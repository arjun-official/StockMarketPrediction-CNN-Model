
"""
Pipeline implementing the problem description:
- Phase 1: Generate chart image dataset for AAPL (primary) and MSFT (secondary)
- Phase 2: Train CNN (ResNet18) to classify {-1,0,1} next-15d move
- Phase 3: Inference CLI demo + RMSE evaluation on held-out tail
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.chart_generator import generate_chart_dataset
from model.helpers import train_classifier, predict_class, rmse


def build_dataset(out_dir: str, start="2010-01-01", end="2015-12-31", window=252, horizon=15):
    df = generate_chart_dataset(
        primary_ticker="AAPL",
        secondary_ticker="MSFT",
        start=start, end=end,
        out_dir=out_dir, window=window, horizon=horizon
    )
    print(f"Generated {len(df)} charts to {out_dir}")
    return df


def split_train_eval(labels_csv: str, eval_fraction: float = 0.2):
    df = pd.read_csv(labels_csv, parse_dates=['date']).set_index('date')
    n = len(df)
    n_eval = max(1, int(n * eval_fraction))
    train_df = df.iloc[:-n_eval]
    eval_df = df.iloc[-n_eval:]
    train_df.to_csv(Path(labels_csv).with_name("labels_train.csv"))
    eval_df.to_csv(Path(labels_csv).with_name("labels_eval.csv"))
    return train_df, eval_df


def evaluate_rmse(model_path: str, labels_csv_eval: str) -> float:
    df = pd.read_csv(labels_csv_eval, parse_dates=['date']).set_index('date')
    preds = []
    actual = []
    for _, row in df.iterrows():
        p = predict_class(model_path, row['image_path'])
        preds.append(p)
        actual.append(int(row['label']))
    score = rmse(np.array(preds, dtype=float), np.array(actual, dtype=float))
    print(f"RMSE on held-out set: {score:.4f}")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="charts", help="Directory to store charts and labels")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2015-12-31")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--model_out", default="chart_cnn.pt")
    args = parser.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)

    # Phase 1
    df = build_dataset(out_dir=str(wd), start=args.start, end=args.end, window=args.window, horizon=args.horizon)
    labels_csv = wd / "labels.csv"

    # Split
    train_df, eval_df = split_train_eval(str(labels_csv))

    # Phase 2
    train_classifier(str(wd / "labels_train.csv"), out_path=args.model_out)

    # Phase 3: Evaluate on held-out tail
    evaluate_rmse(args.model_out, str(wd / "labels_eval.csv"))


if __name__ == "__main__":
    main()
