import unittest

from app import build_structured_prompt, example_value, normalize_field_rows


class LlmFieldConfigTests(unittest.TestCase):
    def test_normalize_field_rows_filters_invalid_and_maps_types(self):
        rows = [
            {
                "name": "is_target",
                "type_label": "布尔",
                "requirement": "是否属于目标类别",
                "example": "true",
                "required": True,
                "enum_options": "",
            },
            {"name": "bad.name", "type_label": "文本", "requirement": "invalid", "example": "", "required": True, "enum_options": ""},
            {"name": "level", "type_label": "枚举", "requirement": "相关程度", "example": "高", "required": False, "enum_options": "高,中,低"},
        ]

        fields = normalize_field_rows(rows)

        self.assertEqual([field["name"] for field in fields], ["is_target", "level"])
        self.assertEqual(fields[0]["type"], "boolean")
        self.assertEqual(fields[1]["type"], "enum")
        self.assertIn("可选值：高,中,低", fields[1]["requirement"])

    def test_build_structured_prompt_uses_user_examples(self):
        fields = normalize_field_rows(
            [
                {"name": "confidence", "type_label": "数字", "requirement": "置信度", "example": "0.91", "required": True, "enum_options": ""},
                {"name": "count", "type_label": "整数", "requirement": "数量", "example": "3", "required": True, "enum_options": ""},
            ]
        )

        prompt = build_structured_prompt("判断文本", fields)

        self.assertIn('"confidence": 0.91', prompt)
        self.assertIn('"count": 3', prompt)
        self.assertEqual(example_value("boolean", "是"), True)


if __name__ == "__main__":
    unittest.main()
