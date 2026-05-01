import unittest

from judgelab.llm import extract_json, normalize_bool


class LlmTest(unittest.TestCase):
    def test_extract_json_from_markdown_block(self):
        data = extract_json('```json\n{"is_target": true, "core_reason": "命中"}\n```')

        self.assertEqual(data["is_target"], True)
        self.assertEqual(data["core_reason"], "命中")

    def test_normalize_bool(self):
        self.assertTrue(normalize_bool("相关"))
        self.assertFalse(normalize_bool("不相关"))


if __name__ == "__main__":
    unittest.main()

