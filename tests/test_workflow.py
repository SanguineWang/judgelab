import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from judgelab.importer import ImportOptions, import_excel_files
from judgelab.workflow import (
    clear_downstream_assets,
    count_filtered_records,
    create_filtered_sample,
    create_random_sample,
    create_splits,
    create_stratified_sample,
    label_distribution,
    get_asset_stats,
    preview_table,
    replace_records_table,
)
from judgelab.workspace import WorkspaceStore


def make_xlsx(path: Path, headers, rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    workbook.save(path)


class WorkflowTest(unittest.TestCase):
    def test_random_sample_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("流程测试")
            file_path = root / "data.xlsx"
            make_xlsx(file_path, ["摘要", "标签"], [[f"text-{i}", "是" if i % 2 == 0 else "否"] for i in range(20)])
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())

            result = create_random_sample(store, dataset.dataset_id, sample_size=5, seed=1)
            preview = preview_table(store, dataset.dataset_id, "sampled_records", limit=10)

        self.assertEqual(result.row_count, 5)
        self.assertEqual(len(preview), 5)

    def test_split_sampled_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("划分测试")
            file_path = root / "data.xlsx"
            make_xlsx(file_path, ["摘要", "标签"], [[f"text-{i}", "是" if i < 10 else "否"] for i in range(20)])
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())
            create_random_sample(store, dataset.dataset_id, sample_size=20, seed=1)

            split_result = create_splits(store, dataset.dataset_id, source_table="sampled_records", label_column="标签")
            stats = get_asset_stats(store, dataset.dataset_id)

        self.assertEqual(split_result.total_count, 20)
        self.assertEqual(stats["train"], 16)
        self.assertEqual(stats["val"], 2)
        self.assertEqual(stats["test"], 2)

    def test_keyword_filtered_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("筛选抽样测试")
            file_path = root / "data.xlsx"
            make_xlsx(
                file_path,
                ["标题", "摘要", "标签"],
                [
                    ["低空物流无人机", "城市配送", "是"],
                    ["农用无人机", "植保巡检", "是"],
                    ["新能源汽车", "电池系统", "否"],
                    ["eVTOL", "低空载人飞行器", "是"],
                ],
            )
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())

            matched = count_filtered_records(store, dataset.dataset_id, ["标题", "摘要"], ["无人机", "eVTOL"], keyword_mode="any")
            result = create_filtered_sample(store, dataset.dataset_id, ["标题", "摘要"], ["无人机", "eVTOL"], sample_size=10, seed=1)
            preview = preview_table(store, dataset.dataset_id, "sampled_records", limit=10)

        self.assertEqual(matched, 3)
        self.assertEqual(result.row_count, 3)
        self.assertTrue(all(row["__sample_source"] == "keyword_filter" for row in preview))

    def test_keyword_filter_all_keywords_across_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("全部关键词测试")
            file_path = root / "data.xlsx"
            make_xlsx(
                file_path,
                ["标题", "摘要"],
                [
                    ["无人机", "低空物流配送"],
                    ["无人机", "植保巡检"],
                    ["低空经济", "物流园区"],
                ],
            )
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())

            matched = count_filtered_records(store, dataset.dataset_id, ["标题", "摘要"], ["无人机", "物流"], keyword_mode="all")

        self.assertEqual(matched, 1)

    def test_stratified_sample_marks_source_without_helper_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("分层来源测试")
            file_path = root / "data.xlsx"
            make_xlsx(file_path, ["摘要", "地区"], [[f"text-{i}", "北京" if i < 5 else "上海"] for i in range(10)])
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())

            result = create_stratified_sample(store, dataset.dataset_id, "地区", sample_size=4, seed=1)
            preview = preview_table(store, dataset.dataset_id, "sampled_records", limit=10)

        self.assertEqual(result.row_count, 4)
        self.assertTrue(all(row["__sample_source"] == "stratified" for row in preview))
        self.assertNotIn("group_value", preview[0])
        self.assertNotIn("sample_count", preview[0])

    def test_replace_records_table_and_label_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(Path(tmp) / "workspace")
            dataset = store.create_dataset("标注测试")

            result = replace_records_table(
                store,
                dataset.dataset_id,
                "labeled_records",
                [{"文本": "a", "标签": "是"}, {"文本": "b", "标签": "否"}, {"文本": "c", "标签": "是"}],
            )
            distribution = label_distribution(store, dataset.dataset_id, "labeled_records", "标签")

        self.assertEqual(result.row_count, 3)
        self.assertEqual(distribution, {"是": 2, "否": 1})

    def test_resampling_clears_downstream_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("重跑抽样测试")
            file_path = root / "data.xlsx"
            make_xlsx(file_path, ["摘要", "标签"], [[f"text-{i}", "是" if i < 10 else "否"] for i in range(20)])
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())
            create_random_sample(store, dataset.dataset_id, sample_size=20, seed=1)
            replace_records_table(store, dataset.dataset_id, "labeled_records", [{"摘要": "old", "标签": "是"}])
            replace_records_table(store, dataset.dataset_id, "reviewed_records", [{"摘要": "old", "标签": "是"}])
            create_splits(store, dataset.dataset_id, source_table="sampled_records", label_column="标签")

            create_random_sample(store, dataset.dataset_id, sample_size=5, seed=2)
            stats = get_asset_stats(store, dataset.dataset_id)

        self.assertEqual(stats["sampled"], 5)
        self.assertEqual(stats["labeled"], 0)
        self.assertEqual(stats["reviewed"], 0)
        self.assertEqual(stats["train"], 0)
        self.assertEqual(stats["val"], 0)
        self.assertEqual(stats["test"], 0)

    def test_clearing_reviewed_clears_split_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "workspace")
            dataset = store.create_dataset("质检重跑测试")
            file_path = root / "data.xlsx"
            make_xlsx(file_path, ["摘要", "标签"], [[f"text-{i}", "是" if i < 10 else "否"] for i in range(20)])
            import_excel_files(store, dataset.dataset_id, [file_path], ImportOptions())
            create_random_sample(store, dataset.dataset_id, sample_size=20, seed=1)
            create_splits(store, dataset.dataset_id, source_table="sampled_records", label_column="标签")

            clear_downstream_assets(store, dataset.dataset_id, "reviewed")
            stats = get_asset_stats(store, dataset.dataset_id)

        self.assertEqual(stats["sampled"], 20)
        self.assertEqual(stats["train"], 0)
        self.assertEqual(stats["val"], 0)
        self.assertEqual(stats["test"], 0)


if __name__ == "__main__":
    unittest.main()
