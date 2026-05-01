import unittest

from judgelab.sampling import proportional_allocations, random_sample, stratified_sample


class SamplingTest(unittest.TestCase):
    def test_random_sample_is_reproducible(self):
        records = [{"id": index} for index in range(10)]

        first = random_sample(records, 4, seed=7)
        second = random_sample(records, 4, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_stratified_sample_preserves_groups(self):
        records = [{"id": index, "group": "A" if index < 8 else "B"} for index in range(10)]

        sampled = stratified_sample(records, "group", 5, seed=1)
        groups = {row["group"] for row in sampled}

        self.assertEqual(len(sampled), 5)
        self.assertEqual(groups, {"A", "B"})

    def test_proportional_allocations_sum_to_total(self):
        allocations = proportional_allocations({"A": 8, "B": 2}, 5)

        self.assertEqual(sum(allocations.values()), 5)
        self.assertGreaterEqual(allocations["B"], 1)


if __name__ == "__main__":
    unittest.main()

