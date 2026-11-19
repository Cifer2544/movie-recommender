"""探索性数据分析：生成报告与演示用图表。

输出（均写入 outputs/figures/，该目录已被 git 忽略）：
    01_rating_distribution.png   评分分布
    02_movie_long_tail.png       电影热度长尾 + 累积覆盖曲线
    03_user_activity.png         用户活跃度分布
    04_genre_popularity.png      题材热度（按评分数计）
另将关键统计量写入 outputs/results/eda_stats.txt。

图表统一使用英文标签：matplotlib 默认字体不含中文字形，避免出现方块。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config
from .data_loader import load_all
from .split import leave_one_out_split

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BLUE = "#4C72B0"
ORANGE = "#DD8452"


def plot_rating_distribution(ratings: pd.DataFrame, path) -> pd.DataFrame:
    counts = ratings["Rating"].value_counts().sort_index()
    share = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(counts.index.astype(str), counts.values, color=BLUE, width=0.62)
    ax.bar_label(
        bars,
        labels=[f"{v / 1000:.0f}k\n{s:.1%}" for v, s in zip(counts.values, share.values)],
        fontsize=7,
        padding=2,
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_title(
        f"Rating distribution (n={len(ratings):,}, mean={ratings['Rating'].mean():.2f})"
    )
    ax.set_ylim(0, counts.max() * 1.22)
    fig.savefig(path)
    plt.close(fig)

    return pd.DataFrame({"rating": counts.index, "count": counts.values, "share": share.values})


def plot_movie_long_tail(ratings: pd.DataFrame, path) -> dict:
    counts = ratings.groupby("MovieID").size().sort_values(ascending=False)
    rank = np.arange(1, len(counts) + 1)
    cumulative = counts.cumsum() / counts.sum()

    top10 = int(len(counts) * 0.1)
    coverage_top10 = cumulative.iloc[top10 - 1]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(rank, counts.to_numpy(), color=BLUE, lw=1.2, label="ratings per movie")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Movie rank (log)")
    ax.set_ylabel("Number of ratings (log)")
    ax.set_title(f"Long tail of movie popularity ({len(counts):,} movies)")
    ax.legend(loc="upper right", frameon=False)

    ax2 = ax.twinx()
    ax2.plot(rank, cumulative.to_numpy(), color=ORANGE, lw=1.2, ls="--")
    ax2.set_ylabel("Cumulative share of ratings", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.set_ylim(0, 1.02)
    ax2.grid(False)

    ax.annotate(
        f"top 10% movies\n= {coverage_top10:.1%} of ratings",
        xy=(top10, counts.iloc[top10 - 1]),
        xytext=(3, 4),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )
    fig.savefig(path)
    plt.close(fig)

    return {
        "movies": len(counts),
        "top10_share": float(coverage_top10),
        "median_ratings_per_movie": int(counts.median()),
        "movies_with_lt10_ratings": int((counts < 10).sum()),
    }


def plot_user_activity(ratings: pd.DataFrame, path) -> dict:
    activity = ratings.groupby("UserID").size()

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(activity.to_numpy(), bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Number of users")
    ax.set_title(f"User activity (n={len(activity):,} users)")
    ax.axvline(activity.median(), color=ORANGE, ls="--", lw=1.2)
    ax.annotate(
        f"median = {activity.median():.0f}",
        xy=(activity.median(), ax.get_ylim()[1] * 0.92),
        xytext=(8, 0),
        textcoords="offset points",
        color=ORANGE,
        fontsize=8,
    )
    fig.savefig(path)
    plt.close(fig)

    return {
        "users": len(activity),
        "min": int(activity.min()),
        "median": float(activity.median()),
        "mean": float(activity.mean()),
        "max": int(activity.max()),
    }


def plot_genre_popularity(ratings: pd.DataFrame, movies: pd.DataFrame, path) -> pd.Series:
    genre_counts = (
        ratings[["MovieID"]]
        .merge(movies[["MovieID", "GenreList"]], on="MovieID", how="left")
        .explode("GenreList")
        .groupby("GenreList")
        .size()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    bars = ax.barh(genre_counts.index, genre_counts.values, color=BLUE)
    ax.bar_label(
        bars, labels=[f"{v / 1000:.0f}k" for v in genre_counts.values], fontsize=7, padding=2
    )
    ax.set_xlabel("Number of ratings")
    ax.set_title("Genre popularity (a movie counts in every genre it has)")
    ax.set_xlim(0, genre_counts.max() * 1.15)
    fig.savefig(path)
    plt.close(fig)

    return genre_counts


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ratings, movies, users = load_all()

    rating_dist = plot_rating_distribution(ratings, config.FIGURES_DIR / "01_rating_distribution.png")
    tail = plot_movie_long_tail(ratings, config.FIGURES_DIR / "02_movie_long_tail.png")
    activity = plot_user_activity(ratings, config.FIGURES_DIR / "03_user_activity.png")
    genre = plot_genre_popularity(ratings, movies, config.FIGURES_DIR / "04_genre_popularity.png")

    splits = leave_one_out_split(ratings)
    train = splits["train"]

    lines = ["== 评分分布 ==", rating_dist.to_string(index=False), ""]
    lines += ["== 电影长尾 ==", " ".join(f"{k}={v}" for k, v in tail.items()), ""]
    lines += ["== 用户活跃度 ==", " ".join(f"{k}={v}" for k, v in activity.items()), ""]
    lines += ["== 题材热度 Top5 ==", genre.sort_values(ascending=False).head().to_string(), ""]
    lines += [
        "== 划分后的分布漂移检查 ==",
        f"train 均分 {train['Rating'].mean():.4f} / 全体 {ratings['Rating'].mean():.4f}",
        f"train 正反馈比例 {(train['Rating'] >= config.POSITIVE_RATING_THRESHOLD).mean():.4f}",
        f"test 正反馈比例 {(splits['test']['Rating'] >= config.POSITIVE_RATING_THRESHOLD).mean():.4f}",
    ]
    text = "\n".join(lines)

    print(text)
    print(f"\n图表: {config.FIGURES_DIR.relative_to(config.ROOT)}\\0[1-4]_*.png")
    (config.RESULTS_DIR / "eda_stats.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
