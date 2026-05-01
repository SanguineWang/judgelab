import tempfile
import unittest
import zipfile
from pathlib import Path

from openpyxl import Workbook

from judgelab.importer import ImportOptions, import_excel_files, preflight_excel_files
from judgelab.workspace import WorkspaceStore


def make_xlsx(path: Path, headers, rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def corrupt_sheet_dimension(path: Path, dimension: str = "A1"):
    replacement = f'<dimension ref="{dimension}"/>'
    temp_path = path.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "xl/worksheets/sheet1.xml":
                text = data.decode("utf-8")
                start = text.find("<dimension ")
                end = text.find("/>", start)
                if start != -1 and end != -1:
                    text = text[:start] + replacement + text[end + 2 :]
                    data = text.encode("utf-8")
            target.writestr(info, data)
    temp_path.replace(path)


class WorkspaceImporterTest(unittest.TestCase):
    def test_create_and_list_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(Path(tmp))
            dataset = store.create_dataset("低空经济专利", "测试数据集")

            datasets = store.list_datasets()

        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0].dataset_id, dataset.dataset_id)
        self.assertEqual(datasets[0].name, "低空经济专利")

    def test_preflight_detects_header_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok = root / "ok.xlsx"
            reordered = root / "reordered.xlsx"
            missing = root / "missing.xlsx"
            extra = root / "extra.xlsx"
            make_xlsx(ok, ["摘要", "年份", "地区"], [["a", 2024, "北京"]])
            make_xlsx(reordered, ["地区", "摘要", "年份"], [["上海", "b", 2025]])
            make_xlsx(missing, ["摘要", "年份"], [["c", 2025]])
            make_xlsx(extra, ["摘要", "年份", "地区", "行业"], [["d", 2025, "深圳", "低空"]])

            report = preflight_excel_files([ok, reordered, missing, extra])

        statuses = {item.file_name: item.status for item in report.files}
        self.assertEqual(statuses["ok.xlsx"], "matched")
        self.assertEqual(statuses["reordered.xlsx"], "reordered")
        self.assertEqual(statuses["missing.xlsx"], "missing_columns")
        self.assertEqual(statuses["extra.xlsx"], "extra_columns")
        self.assertFalse(report.can_import_strict)

    def test_import_multiple_excels_to_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("导入测试")
            first = root / "first.xlsx"
            second = root / "second.xlsx"
            make_xlsx(first, ["摘要", "年份"], [["a", 2024], ["b", 2025]])
            make_xlsx(second, ["年份", "摘要"], [[2026, "c"]])

            summary = import_excel_files(store, dataset.dataset_id, [first, second], ImportOptions())
            stats = store.get_dataset_stats(dataset.dataset_id)
            preview = store.preview_raw_records(dataset.dataset_id, limit=10)

        self.assertEqual(summary.imported_rows, 3)
        self.assertEqual(stats.row_count, 3)
        self.assertEqual(stats.column_count, 2)
        self.assertEqual(preview[0]["摘要"], "a")
        self.assertEqual(preview[2]["摘要"], "c")

    def test_import_excel_with_wrong_sheet_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("异常维度测试")
            file_path = root / "bad_dimension.xlsx"
            make_xlsx(file_path, ["序号", "摘要"], [[1, "a"], [2, "b"]])
            corrupt_sheet_dimension(file_path, "A1")

            report = preflight_excel_files([file_path])
            summary = import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())
            stats = store.get_dataset_stats(dataset.dataset_id)

        self.assertEqual(report.reference_headers, ["序号", "摘要"])
        self.assertEqual(report.total_rows, 2)
        self.assertEqual(summary.imported_rows, 2)
        self.assertEqual(stats.row_count, 2)

    def test_clear_dataset_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("清空测试")
            file_path = root / "data.xlsx"
            make_xlsx(file_path, ["摘要"], [["a"], ["b"]])
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())

            store.clear_dataset(dataset.dataset_id, scope="all")
            stats = store.get_dataset_stats(dataset.dataset_id)

        self.assertEqual(stats.row_count, 0)
        self.assertEqual(stats.column_count, 0)


if __name__ == "__main__":
    unittest.main()
