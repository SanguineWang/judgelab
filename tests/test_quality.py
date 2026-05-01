import unittest

from judgelab.quality import cohen_kappa, label_distribution, low_confidence_rows, missing_label_rows


class QualityTest(unittest.TestCase):
    def test_label_distribution_counts_missing(self):
        records = [{"label": "是"}, {"label": "否"}, {"label": ""}, {"label": None}]

        distribution = label_distribution(records, "label")

        self.assertEqual(distribution["是"], 1)
        self.assertEqual(distribution["否"], 1)
        self.assertEqual(distribution["未标注"], 2)

    def test_missing_label_rows(self):
        records = [{"label": "是"}, {"label": ""}, {"label": None}]

        self.assertEqual(missing_label_rows(records, "label"), [1, 2])

    def test_low_confidence_rows(self):
        records = [{"score": 0.9}, {"score": "0.5"}, {"score": "bad"}]

        self.assertEqual(low_confidence_rows(records, "score", 0.7), [1])

    def test_cohen_kappa(self):
        kappa = cohen_kappa(["是", "是", "否", "否"], ["是", "否", "否", "否"])

        self.assertAlmostEqual(kappa, 0.5)


if __name__ == "__main__":
    unittest.main()

