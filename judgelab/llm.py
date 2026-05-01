from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence


DEFAULT_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


@dataclass(frozen=True)
class LlmJudgeConfig:
    api_key: str
    api_url: str = DEFAULT_API_URL
    model: str = "glm-5.1"
    system_prompt: str = ""
    text_column: str = "摘要"
    label_column: str = "LLM判定"
    reason_column: str = "LLM理由"
    raw_column: str = "LLM_JSON"
    error_column: str = "LLM_ERROR"
    boolean_key: str = "is_target"
    reason_key: str = "core_reason"
    limit: int | None = None
    timeout: int = 60
    sleep_seconds: float = 0.0


def build_user_prompt(text: str) -> str:
    return f"""待判定文本：
<<<开始>>>
{text}
<<<结束>>>
"""


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM 返回的 JSON 顶层不是对象")
    return data


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "属于", "相关", "正样本"}
    return False


def call_chat_json(config: LlmJudgeConfig, text: str) -> Dict[str, Any]:
    if not config.api_key:
        raise ValueError("缺少 API Key")
    if not config.system_prompt.strip():
        raise ValueError("缺少系统 Prompt")

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": build_user_prompt(text)},
        ],
        "temperature": 0.1,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        config.api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"请求失败: {error.reason}") from error

    response_json = json.loads(response_body)
    content = response_json["choices"][0]["message"]["content"]
    return extract_json(content)


def judge_records(
    records: Sequence[Dict[str, Any]],
    config: LlmJudgeConfig,
    log: Callable[[str], None] | None = None,
) -> List[Dict[str, Any]]:
    logger = log or (lambda message: None)
    output: List[Dict[str, Any]] = []
    processed = 0

    for index, record in enumerate(records, start=1):
        merged = dict(record)
        text = str(record.get(config.text_column, "") or "").strip()
        if not text:
            merged[config.error_column] = "文本为空"
            output.append(merged)
            continue
        if config.limit is not None and processed >= config.limit:
            output.append(merged)
            continue

        try:
            logger(f"第 {index} 行开始 LLM 判定")
            result = call_chat_json(config, text)
            label = normalize_bool(result.get(config.boolean_key))
            reason = str(result.get(config.reason_key) or result.get("reason") or "").strip()
            merged[config.label_column] = label
            merged[config.reason_column] = reason
            merged[config.raw_column] = json.dumps(result, ensure_ascii=False)
            merged[config.error_column] = ""
            processed += 1
            logger(f"第 {index} 行完成：{label}")
        except Exception as exc:
            merged[config.error_column] = str(exc)
            logger(f"第 {index} 行失败：{exc}")

        output.append(merged)
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    return output

