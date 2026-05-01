import tempfile
import unittest
from pathlib import Path

from judgelab.data_io import read_table, write_table


class DataIoTest(unittest.TestCase):
    def test_write_and_read_xlsx(self):
        rows = [{"摘要": "文本1", "标签": "是"}, {"摘要": "文本2", "标签": "否"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.xlsx"

            write_table(path, rows, ["摘要", "标签"])
            table = read_table(path)

        self.assertEqual(table.headers, ["摘要", "标签"])
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.rows[0]["摘要"], "文本1")

    def test_write_and_read_csv(self):
        rows = [{"text": "a", "label": "1"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"

            write_table(path, rows, ["text", "label"])
            table = read_table(path)

        self.assertEqual(table.headers, ["text", "label"])
        self.assertEqual(table.rows[0]["label"], "1")


if __name__ == "__main__":
    unittest.main()

