import pandas as pd

from src.federated.final_validation import collect
from src.final_report import utility_table


def test_collect_final_includes_precision_metrics():
    rows = collect()
    for column in ["test_macro_precision", "test_weighted_precision", "test_weighted_recall"]:
        assert column in rows.columns, f"missing {column} in final confirmation rows"


def test_utility_table_includes_precision_and_recall_summary_columns():
    table = utility_table(pd.DataFrame([{
        "arm": "demo_arm",
        "condition": "clean",
        "test_accuracy": 0.7,
        "test_macro_precision": 0.2,
        "test_macro_recall": 0.3,
        "test_macro_f1": 0.25,
        "test_weighted_precision": 0.65,
        "test_weighted_recall": 0.7,
        "test_weighted_f1": 0.68,
        "test_macro_auroc": 0.8,
    }]))
    for column in ["macro_precision_mean", "macro_recall_mean",
                   "weighted_precision_mean", "weighted_recall_mean"]:
        assert column in table.columns, f"missing {column} in final utility table"
