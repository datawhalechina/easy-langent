import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import data_manager


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        for attribute, value in (("GLOBAL_DF", None), ("GLOBAL_FILENAME", "test.csv")):
            patcher = patch.object(data_manager, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_mixed_columns_after_removing_an_empty_csv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("amount,category\n0,a\n,\n1,b\n-1,c\n", encoding="utf-8")
            loaded, message = data_manager.load_csv_file(str(path))

        self.assertTrue(loaded, message)
        self.assertEqual(data_manager.get_dataframe().index.tolist(), [0, 2, 3])
        for columns in (("amount", "category"), ("category", "amount")):
            with self.subTest(columns=columns):
                self.assertEqual(data_manager.calculate_correlation(*columns), -0.5)

    def test_mixed_columns_preserve_nondefault_and_duplicate_row_labels(self):
        for index in ([10, 20, 30], [5, 5, 7]):
            with self.subTest(index=index):
                data_manager.GLOBAL_DF = pd.DataFrame(
                    {"amount": [0, 1, -1], "category": ["a", "b", "c"]},
                    index=index,
                )
                self.assertEqual(data_manager.calculate_correlation("amount", "category"), -0.5)
                self.assertEqual(data_manager.get_dataframe().index.tolist(), index)

    def test_numeric_columns_keep_their_row_pairing(self):
        data_manager.GLOBAL_DF = pd.DataFrame(
            {"amount": [0, 1, -1], "rank": [0, 1, 2]}, index=[0, 2, 3]
        )
        self.assertEqual(data_manager.calculate_correlation("amount", "rank"), -0.5)

    def test_categorical_columns_keep_their_row_pairing(self):
        data_manager.GLOBAL_DF = pd.DataFrame(
            {"first": ["a", "b", "a"], "second": ["x", "x", "y"]}, index=[5, 5, 7]
        )
        self.assertEqual(data_manager.calculate_correlation("first", "second"), -0.5)


if __name__ == "__main__":
    unittest.main()
