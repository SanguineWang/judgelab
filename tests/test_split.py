import unittest

from judgelab.split import split_counts, stratified_train_val_test_split


class SplitTest(unittest.TestCase):
    def test_split_counts_sum_to_total(self):
        self.assertEqual(sum(split_counts(10, 0.8, 0.1, 0.1)), 10)
        self.assertEqual(split_counts(3, 0.8, 0.1, 0.1), (1, 1, 1))

    def test_stratified_split_keeps_labels(self):
        records = [{"id": index, "label": "是" if index < 10 else "否"} for index in range(20)]

        split_data = stratified_train_val_test_split(records, "label", 0.8, 0.1, 0.1, seed=1)

        self.assertEqual(sum(len(part) for part in split_data.values()), 20)
        for name in ["train", "val", "test"]:
            labels = {row["label"] for row in split_data[name]}
            self.assertEqual(labels, {"是", "否"})


if __name__ == "__main__":
    unittest.main()

