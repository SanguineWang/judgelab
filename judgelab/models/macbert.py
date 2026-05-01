from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = "hfl/chinese-macbert-base"
    output_dir: Path = Path("projects/default/models/v1")
    text_column: str = "摘要"
    label_column: str = "标签"
    max_length: int = 256
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


@dataclass(frozen=True)
class PredictConfig:
    model_dir: Path
    text_column: str = "摘要"
    max_length: int = 256
    batch_size: int = 16


def normalize_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是", "属于", "相关", "正样本"}:
        return 1
    if text in {"false", "0", "no", "n", "否", "不属于", "不相关", "负样本"}:
        return 0
    raise ValueError(f"无法识别二分类标签: {value}")


def build_samples(records: Sequence[Dict[str, Any]], text_column: str, label_column: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        text = str(record.get(text_column, "") or "").strip()
        label_value = record.get(label_column)
        if not text or label_value in (None, ""):
            continue
        samples.append({"index": index, "text": text, "label": normalize_binary_label(label_value)})
    return samples


def compute_metrics(predictions: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    total = len(labels)
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train_classifier(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    config: TrainConfig,
    log: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule = _load_training_deps()
    logger = log or (lambda message: None)

    train_samples = build_samples(train_records, config.text_column, config.label_column)
    val_samples = build_samples(val_records, config.text_column, config.label_column)
    test_samples = build_samples(test_records, config.text_column, config.label_column)
    if len(train_samples) < 2:
        raise ValueError("训练样本少于 2 条，无法训练 MacBERT。")
    labels = {sample["label"] for sample in train_samples}
    if labels != {0, 1}:
        raise ValueError("训练集必须同时包含正样本和负样本。")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model, num_labels=2)
    device = get_device(torch)
    model.to(device)

    dataset_cls = _dataset_class(Dataset, tokenizer, config.max_length)
    train_loader = DataLoader(dataset_cls(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_cls(val_samples), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_cls(test_samples), batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = get_linear_schedule(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_val_metrics: Dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 10 == 0 or step == len(train_loader):
                logger(f"epoch {epoch}/{config.epochs}, step {step}/{len(train_loader)}, train_loss={total_loss / step:.4f}")

        val_metrics = evaluate(model, val_loader, device, torch)
        logger(
            "epoch "
            f"{epoch} eval: accuracy={val_metrics['accuracy']:.4f}, "
            f"precision={val_metrics['precision']:.4f}, recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_val_metrics = val_metrics
            model.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
            (config.output_dir / "label_map.json").write_text(
                json.dumps({"0": "负样本", "1": "正样本"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    test_metrics = evaluate(model, test_loader, device, torch) if test_samples else {}
    result = {
        "model_dir": str(config.output_dir),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    (config.output_dir / "training_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def predict_records(records: Sequence[Dict[str, Any]], config: PredictConfig) -> List[Dict[str, Any]]:
    torch, AutoModelForSequenceClassification, AutoTokenizer = _load_prediction_deps()
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir)
    device = get_device(torch)
    model.to(device)
    model.eval()

    texts = [str(record.get(config.text_column, "") or "").strip() for record in records]
    predictions = predict_texts(texts, tokenizer, model, device, torch, config.max_length, config.batch_size)
    output: List[Dict[str, Any]] = []
    for record, prediction in zip(records, predictions):
        merged = dict(record)
        merged["预测标签"] = "正样本" if prediction["label"] == 1 else "负样本"
        merged["正样本概率"] = prediction["positive_probability"]
        merged["预测置信度"] = prediction["confidence"]
        output.append(merged)
    return output


def predict_texts(texts: Sequence[str], tokenizer: Any, model: Any, device: Any, torch: Any, max_length: int, batch_size: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(batch_texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
        for pred_id, prob_row in zip(pred_ids.cpu().tolist(), probs.cpu().tolist()):
            results.append(
                {
                    "label": int(pred_id),
                    "positive_probability": float(prob_row[1]),
                    "confidence": float(prob_row[pred_id]),
                }
            )
    return results


def evaluate(model: Any, dataloader: Any, device: Any, torch: Any) -> Dict[str, float]:
    if len(dataloader) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "loss": 0.0}
    model.eval()
    predictions: List[int] = []
    labels: List[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = compute_metrics(predictions, labels)
    metrics["loss"] = total_loss / max(1, len(dataloader))
    return metrics


def get_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dataset_class(dataset_base: Any, tokenizer: Any, max_length: int) -> Any:
    class TextDataset(dataset_base):
        def __init__(self, samples: Sequence[Dict[str, Any]]) -> None:
            self.samples = list(samples)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            sample = self.samples[index]
            encoded = tokenizer(sample["text"], max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = _torch().tensor(sample["label"], dtype=_torch().long)
            return item

    return TextDataset


def _torch() -> Any:
    import torch

    return torch


def _load_training_deps() -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("缺少 MacBERT 训练依赖，请安装 requirements.txt 中的 torch 和 transformers。") from exc
    return torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def _load_prediction_deps() -> Any:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 MacBERT 预测依赖，请安装 requirements.txt 中的 torch 和 transformers。") from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer

