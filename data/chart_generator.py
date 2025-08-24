
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def _download(ticker: str, start: str, end: str) -> pd.Series:
    tkr = yf.Ticker(ticker)
    df = tkr.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker} between {start} and {end}")
    return df["Close"].asfreq("B").ffill()


def _rolling_windows(series: pd.Series, window: int) -> List[pd.DatetimeIndex]:
    # builds list of end-exclusive windows: [i-window, i)
    idx = series.dropna().index
    windows = []
    for i in range(window, len(idx)):
        w = idx[i-window:i]
        windows.append(w)
    return windows


def _cumret(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] - 1.0


def _label_future_move(series: pd.Series, horizon: int = 15) -> int:
    # label based on sign of forward pct change over 'horizon' business days
    if len(series) < horizon + 1:
        return 0
    ret = series.iloc[horizon] / series.iloc[0] - 1.0
    if ret > 0:
        return 1
    if ret < 0:
        return -1
    return 0


def generate_chart_dataset(
    primary_ticker: str = "AAPL",
    secondary_ticker: str = "MSFT",
    start: str = "2010-01-01",
    end: str = "2015-12-31",
    out_dir: str = "charts",
    window: int = 252,
    horizon: int = 15,
) -> pd.DataFrame:
    """
    Generates standardized chart images for rolling 1-year windows and CSV labels.
    Returns a DataFrame with columns: ['image_path','date','label']
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = _download(primary_ticker, start, end)
    b = _download(secondary_ticker, start, end)

    # Align to common business day index
    idx = a.index.union(b.index).unique()
    a = a.reindex(idx).ffill()
    b = b.reindex(idx).ffill()

    windows = _rolling_windows(a, window)

    # First pass: compute global y-lims for cumret across both tickers, all windows
    ymin = np.inf
    ymax = -np.inf
    a_windows = []
    b_windows = []
    for w in windows:
        a_w = _cumret(a.loc[w])
        b_w = _cumret(b.loc[w])
        a_windows.append(a_w)
        b_windows.append(b_w)
        ymin = min(ymin, a_w.min(), b_w.min())
        ymax = max(ymax, a_w.max(), b_w.max())

    # Prepare labels based on future move of primary ticker after the window end
    labels = []
    records = []

    for i, w in enumerate(windows):
        ref_date = w[-1]  # last date in window
        # Label from next horizon days of primary ticker
        future_slice = a.loc[ref_date:]
        label = _label_future_move(future_slice, horizon=horizon)

        fig = plt.figure(figsize=(3.2, 3.2), dpi=100)  # ~320x320 px
        ax = plt.gca()
        # Style per spec
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot cumulative returns
        ax.plot(a_windows[i].values, linewidth=1.5, color='white')
        ax.plot(b_windows[i].values, linewidth=1.5, color='white', alpha=0.5)

        # Visual parity: fixed limits across dataset
        ax.set_xlim(0, len(a_windows[i]) - 1)
        ax.set_ylim(ymin, ymax)

        # Save
        fname = f"{ref_date.date()}.png"
        fpath = out_dir / fname
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        records.append({"image_path": str(fpath), "date": ref_date, "label": label})

    df = pd.DataFrame.from_records(records).set_index("date")
    labels_csv = out_dir / "labels.csv"
    df.to_csv(labels_csv)
    return df
