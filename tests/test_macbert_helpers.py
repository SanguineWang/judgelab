import json
import tempfile
import unittest
from pathlib import Path

from judgelab.models.macbert import build_samples, compute_metrics, normalize_binary_label


class MacbertHelpersTest(unittest.TestCase):
    def test_normalize_binary_label(self):
        self.assertEqual(normalize_binary_label("是"), 1)
        self.assertEqual(normalize_binary_label("不相关"), 0)
        self.assertEqual(normalize_binary_label(True), 1)

    def test_build_samples_skips_empty_text(self):
        records = [{"text": "hello", "label": "是"}, {"text": "", "label": "否"}]

        samples = build_samples(records, "text", "label")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 1)

    def test_compute_metrics(self):
        metrics = compute_metrics([1, 0, 1, 0], [1, 0, 0, 0])

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["support"], 4)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)

    def test_training_result_supports_evaluation_details_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "v1"
            model_dir.mkdir(parents=True)
            detail_path = model_dir / "evaluation_details.json"
            detail_path.write_text(
                json.dumps(
                    {
                        "validation": [{"text": "a", "is_correct": True}],
                        "test": [{"text": "b", "is_correct": False}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = json.loads(detail_path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["validation"]), 1)
        self.assertFalse(payload["test"][0]["is_correct"])


if __name__ == "__main__":
    unittest.main()
