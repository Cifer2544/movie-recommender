"""特征编码与相关性标签定义。

本模块只提供"从原始表得到编码后特征"的纯函数，不做任何跨集合统计。
凡是需要统计量的特征（热度、均分、用户活跃度）必须在 src/split.py 划分出的
训练集上计算，否则会把验证/测试信息泄漏进训练特征。

相关性标签（relevance label）
---------------------------
ML-1M 的 1-5 星评分无法直接当排序标签，本项目的转换口径为：

    Rating >= 4  ->  正反馈 / relevant（label = 1）
    Rating <  4  ->  未观测（label = 0，可作负样本候选，但不等同于"不喜欢"）

理由：ML-1M 只有显式评分、没有显式负反馈，3 星及以下既可能是"不喜欢"，
也可能只是"没被推荐过所以不存在"，因此不当作明确的负例，只从负样本池中采样。
ItemCF 直接使用原始评分（评分聚合），BPR / LRURec 使用上面的二值化结果。
"""

from __future__ import annotations

import pandas as pd

from . import config

# ---------------------------------------------------------------
# 编码映射
# ---------------------------------------------------------------
GENDER_MAP = {"F": 0, "M": 1}

# Age 是分组编号而非连续值，映射成 0-6 的序数，避免 56 被当成 56 倍于 1
AGE_TO_ORDINAL = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
# 原始编号 -> 区间名，供报告与图表标注使用（不参与建模）
AGE_LABELS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}


def encode_users(users: pd.DataFrame) -> pd.DataFrame:
    """用户画像编码。

    - Gender -> Gender_bin (0/1)
    - Age    -> Age_idx (0-6 序数)
    - Occupation 保留原始 0-20 编号（类别编号，不可当数值大小解读）
    - Zip-code -> Zip_region（首位数字，粗略地区）；原始 zip 有 3439 个唯一值，
      直接 one-hot 会产生大量低频列，不适合建模
    """
    out = users.copy()
    out["Gender_bin"] = out["Gender"].map(GENDER_MAP).astype("int8")
    out["Age_idx"] = out["Age"].map(AGE_TO_ORDINAL).astype("int8")
    out["Occupation"] = out["Occupation"].astype("int8")
    out["Zip_region"] = out["Zip-code"].str[0].astype("int8")
    return out.drop(columns=["Gender", "Zip-code"])


def genre_columns(movies: pd.DataFrame) -> list[str]:
    """返回全部单题材标签（按字母序，共 18 个）。"""
    return sorted({g for items in movies["GenreList"] for g in items})


def encode_genres(movies: pd.DataFrame) -> pd.DataFrame:
    """题材 multi-hot 编码：每个题材一列 0/1。

    不用 one-hot 的理由：Genres 有 301 种组合，one-hot 会产生 301 列且
    绝大多数组合只出现几次；multi-hot 只需 18 列且保留题材共现信息。
    """
    genres = genre_columns(movies)
    # get_dummies(sep="|") 一次拆完 18 个题材，比逐列 apply 快一个数量级
    flags = movies["Genres"].str.get_dummies(sep="|").astype("int8")
    flags.insert(0, "MovieID", movies["MovieID"].to_numpy())
    return flags[["MovieID", *genres]]


def to_implicit(
    ratings: pd.DataFrame, threshold: int = config.POSITIVE_RATING_THRESHOLD
) -> pd.DataFrame:
    """把 1-5 星评分转成隐式正反馈（口径见模块 docstring）。"""
    pos = ratings[ratings["Rating"] >= threshold]
    return pos[["UserID", "MovieID", "Timestamp"]].reset_index(drop=True)


def standardize(series) -> pd.Series:
    """z-score 标准化，仅用于数值特征（embedding 类模型不需要）。"""
    s = pd.Series(series, dtype="float64")
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return s * 0.0
    return (s - s.mean()) / std


if __name__ == "__main__":
    from .data_loader import load_all

    ratings, movies, users = load_all()

    print("== 用户画像编码 ==")
    ue = encode_users(users)
    print(ue.head(3).to_string(index=False))
    print(f"列: {list(ue.columns)}  shape={ue.shape}")

    print("\n== 题材 multi-hot ==")
    me = encode_genres(movies)
    print(f"shape={me.shape}  题材数={me.shape[1] - 1}")
    per_movie = me.iloc[:, 1:].sum(axis=1)
    print(f"每部电影题材数: min={per_movie.min()}  max={per_movie.max()}")

    print("\n== 相关性标签（隐式正反馈） ==")
    pos = to_implicit(ratings)
    print(
        f"原始评分 {len(ratings)} 条 -> 正反馈 {len(pos)} 条 "
        f"({len(pos) / len(ratings):.2%})，阈值={config.POSITIVE_RATING_THRESHOLD}"
    )

    print("\n== 标准化示例 ==")
    activity = ratings.groupby("UserID").size()
    z = standardize(activity)
    print(
        f"用户活跃度 原始均值={activity.mean():.1f}  "
        f"标准化后均值={z.mean():.2e}  标准差={z.std(ddof=0):.3f}"
    )
