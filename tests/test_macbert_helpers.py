import unittest

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


if __name__ == "__main__":
    unittest.main()

