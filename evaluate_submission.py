"""
Local evaluator for bike sharing submissions.

Default usage:
    python evaluate_submission.py

Evaluate a specific submission:
    python evaluate_submission.py --submission submission_xgboost_micro_ensemble.csv

Compare several submissions:
    python evaluate_submission.py --compare "submission*.csv"

The evaluator aligns predictions by ID. If the answer file is the original
UCI hour.csv, its "instant" column is treated as ID.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ANSWERS = PROJECT_DIR / "hour.csv"
DEFAULT_TEST = PROJECT_DIR /  "test.csv"
DEFAULT_SUBMISSION = PROJECT_DIR / "final_optuna_submission.csv"


def find_id_column(df: pd.DataFrame, file_label: str) -> str:
    for column in ("ID", "instant", "id"):
        if column in df.columns:
            return column
    raise ValueError(f"{file_label} must contain an ID-like column: ID or instant")


def load_test_ids(test_path: Path) -> pd.Series:
    test = pd.read_csv(test_path)
    id_col = find_id_column(test, str(test_path))
    ids = test[id_col].astype(int)
    if ids.duplicated().any():
        duplicated = ids[ids.duplicated()].head(10).tolist()
        raise ValueError(f"Duplicate IDs in test file, examples: {duplicated}")
    return ids


def load_truth(answers_path: Path, test_path: Path | None) -> pd.DataFrame:
    answers = pd.read_csv(answers_path)
    id_col = find_id_column(answers, str(answers_path))
    if "cnt" not in answers.columns:
        raise ValueError(f"{answers_path} must contain true target column 'cnt'")

    truth = answers[[id_col, "cnt"]].copy()
    truth.rename(columns={id_col: "ID", "cnt": "true_cnt"}, inplace=True)
    truth["ID"] = truth["ID"].astype(int)
    truth["true_cnt"] = truth["true_cnt"].astype(float)

    if truth["ID"].duplicated().any():
        duplicated = truth.loc[truth["ID"].duplicated(), "ID"].head(10).tolist()
        raise ValueError(f"Duplicate IDs in answers file, examples: {duplicated}")

    if test_path is not None:
        test_ids = load_test_ids(test_path)
        truth = pd.DataFrame({"ID": test_ids}).merge(truth, on="ID", how="left")
        if truth["true_cnt"].isna().any():
            missing = truth.loc[truth["true_cnt"].isna(), "ID"].head(10).tolist()
            raise ValueError(f"Answers file is missing test IDs, examples: {missing}")

    return truth


def load_submission(submission_path: Path) -> pd.DataFrame:
    submission = pd.read_csv(submission_path)
    id_col = find_id_column(submission, str(submission_path))
    if "cnt" not in submission.columns:
        raise ValueError(f"{submission_path} must contain prediction column 'cnt'")

    pred = submission[[id_col, "cnt"]].copy()
    pred.rename(columns={id_col: "ID", "cnt": "pred_cnt"}, inplace=True)
    pred["ID"] = pred["ID"].astype(int)
    pred["pred_cnt"] = pd.to_numeric(pred["pred_cnt"], errors="coerce")

    if pred["ID"].duplicated().any():
        duplicated = pred.loc[pred["ID"].duplicated(), "ID"].head(10).tolist()
        raise ValueError(f"Duplicate IDs in submission, examples: {duplicated}")
    if pred["pred_cnt"].isna().any():
        bad = pred.loc[pred["pred_cnt"].isna(), "ID"].head(10).tolist()
        raise ValueError(f"Non-numeric or missing predictions, examples IDs: {bad}")

    return pred


def evaluate_one(
    submission_path: Path,
    truth: pd.DataFrame,
    strict_ids: bool = True,
) -> tuple[dict, pd.DataFrame]:
    pred = load_submission(submission_path)
    merged = truth.merge(pred, on="ID", how="left", indicator=True)

    missing_mask = merged["_merge"].ne("both")
    if missing_mask.any():
        missing = merged.loc[missing_mask, "ID"].head(10).tolist()
        raise ValueError(f"{submission_path} is missing test IDs, examples: {missing}")

    if strict_ids:
        extra_ids = sorted(set(pred["ID"]) - set(truth["ID"]))
        if extra_ids:
            raise ValueError(f"{submission_path} has extra IDs, examples: {extra_ids[:10]}")

    residual = merged["true_cnt"].to_numpy() - merged["pred_cnt"].to_numpy()
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))

    result = {
        "submission": str(submission_path),
        "rows": int(len(merged)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias_true_minus_pred": bias,
        "pred_min": float(merged["pred_cnt"].min()),
        "pred_max": float(merged["pred_cnt"].max()),
        "true_min": float(merged["true_cnt"].min()),
        "true_max": float(merged["true_cnt"].max()),
    }

    merged.drop(columns="_merge", inplace=True)
    merged["residual_true_minus_pred"] = residual
    merged["abs_error"] = np.abs(residual)
    merged["squared_error"] = residual ** 2
    return result, merged


def bucket_report(merged: pd.DataFrame) -> pd.DataFrame:
    bins = [
        (0, 50, "0-50"),
        (50, 200, "50-200"),
        (200, 500, "200-500"),
        (500, np.inf, "500+"),
    ]
    rows = []
    for low, high, label in bins:
        mask = (merged["true_cnt"] >= low) & (merged["true_cnt"] < high)
        if not mask.any():
            continue
        residual = merged.loc[mask, "residual_true_minus_pred"].to_numpy()
        rows.append(
            {
                "true_cnt_bucket": label,
                "count": int(mask.sum()),
                "mse": float(np.mean(residual ** 2)),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias_true_minus_pred": float(np.mean(residual)),
            }
        )
    return pd.DataFrame(rows)


def print_single_report(result: dict, merged: pd.DataFrame, show_buckets: bool) -> None:
    print("=" * 72)
    print(f"Submission: {result['submission']}")
    print(f"Rows: {result['rows']}")
    print(f"MSE : {result['mse']:.6f}")
    print(f"RMSE: {result['rmse']:.6f}")
    print(f"MAE : {result['mae']:.6f}")
    print(f"Bias(true - pred): {result['bias_true_minus_pred']:.6f}")
    print(f"Pred range: [{result['pred_min']:.3f}, {result['pred_max']:.3f}]")
    print(f"True range: [{result['true_min']:.3f}, {result['true_max']:.3f}]")

    if show_buckets:
        print("\nError by true cnt bucket:")
        print(bucket_report(merged).to_string(index=False))


def compare_submissions(pattern: str, truth: pd.DataFrame, strict_ids: bool) -> pd.DataFrame:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No submissions matched pattern: {pattern}")

    rows = []
    for path in paths:
        try:
            result, _ = evaluate_one(path, truth, strict_ids=strict_ids)
            rows.append(result)
        except Exception as exc:
            rows.append(
                {
                    "submission": str(path),
                    "rows": np.nan,
                    "mse": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias_true_minus_pred": np.nan,
                    "error": str(exc),
                }
            )
    results = pd.DataFrame(rows).sort_values("mse", na_position="last")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bike sharing submission MSE locally.")
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS, help="Path to answer file, default: hour.csv")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Path to test.csv used to choose IDs")
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION, help="Submission CSV to evaluate")
    parser.add_argument("--compare", type=str, default=None, help='Glob pattern to compare, e.g. "submission*.csv"')
    parser.add_argument("--no-strict-ids", action="store_true", help="Allow extra IDs in submission")
    parser.add_argument("--details", type=Path, default=None, help="Optional path to save row-level residuals")
    parser.add_argument("--buckets", action="store_true", help="Print error by true cnt bucket")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = None if str(args.test).lower() in {"none", ""} else args.test
    truth = load_truth(args.answers, test_path)
    strict_ids = not args.no_strict_ids

    if args.compare:
        results = compare_submissions(args.compare, truth, strict_ids)
        print(results.to_string(index=False))
        out_path = PROJECT_DIR / "local_evaluation_results.csv"
        results.to_csv(out_path, index=False)
        print(f"\nSaved comparison to: {out_path}")
        return

    result, merged = evaluate_one(args.submission, truth, strict_ids=strict_ids)
    print_single_report(result, merged, show_buckets=args.buckets)

    if args.details:
        merged.to_csv(args.details, index=False)
        print(f"\nSaved row-level details to: {args.details}")


if __name__ == "__main__":
    main()
