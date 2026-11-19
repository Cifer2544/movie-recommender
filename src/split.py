"""可复现的数据划分：每用户时序留一。

规则
----
每个用户的评分按 (Timestamp, MovieID) 升序稳定排序后：

    最后 1 条        -> test
    倒数第 2 条      -> valid
    其余（>= 18 条） -> train

为什么不用随机切分或全局时序切分
--------------------------------
1. 三个模型（ItemCF / BPR / LRURec）必须共用同一份划分；LRURec 是序列模型，
   随机切分会把"未来行为"泄漏进历史序列。
2. 留一保证每个用户都有 train/valid/test，评测时 6040 个用户全部参与，
   不会出现"测试用户没有任何训练记录"而指标被稀释的情况。
3. 划分本身不含随机性（SEED 只影响下游负采样与模型初始化），可逐行复现。

防泄漏约定
----------
- ItemCF 的 item-item 相似度矩阵、BPR / LRURec 的负采样池、所有统计特征
  （热度、均分、用户活跃度）只允许在 train 上计算。
- valid / test 只参与打分与评估，不进入特征、不进入负采样。
- assert_no_leakage() 做机械校验：三元组互不重叠、每用户名额正确、时间序单调。
"""

from __future__ import annotations

import pandas as pd

from . import config

SPLITS_DIR = config.OUTPUTS_DIR / "splits"
SPLIT_NAMES = ("train", "valid", "test")


def leave_one_out_split(
    ratings: pd.DataFrame,
    n_valid: int = config.N_VALID_PER_USER,
    n_test: int = config.N_TEST_PER_USER,
) -> dict[str, pd.DataFrame]:
    """按用户时序切分为 train / valid / test。"""
    df = ratings.sort_values(
        ["UserID", "Timestamp", "MovieID"], kind="mergesort"
    ).reset_index(drop=True)

    # rank_from_tail: 0 = 该用户最后一条, 1 = 倒数第二条, ...
    df["rank_from_tail"] = df.groupby("UserID", observed=True).cumcount(ascending=False)

    splits = {
        "test": df[df["rank_from_tail"] < n_test],
        "valid": df[
            (df["rank_from_tail"] >= n_test)
            & (df["rank_from_tail"] < n_test + n_valid)
        ],
        "train": df[df["rank_from_tail"] >= n_test + n_valid],
    }
    splits = {
        name: part.drop(columns=["rank_from_tail"]).reset_index(drop=True)
        for name, part in splits.items()
    }

    total = sum(len(p) for p in splits.values())
    assert total == len(ratings), f"划分后行数 {total} 与原始 {len(ratings)} 不一致"
    return splits


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """硬校验：不重叠、名额正确、时间序单调。任一不满足直接抛异常。"""
    train, valid, test = (splits[name] for name in SPLIT_NAMES)

    # 1. 任意两个集合的 (UserID, MovieID) 组合互不重叠
    for name_a, name_b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        idx_a = splits[name_a].set_index(["UserID", "MovieID"]).index
        idx_b = splits[name_b].set_index(["UserID", "MovieID"]).index
        overlap = idx_a.intersection(idx_b)
        assert len(overlap) == 0, (
            f"{name_a} 与 {name_b} 存在 {len(overlap)} 条重叠的 (UserID, MovieID)"
        )

    # 2. 每个用户的 valid / test 各恰好 1 条，且覆盖与 train 相同的用户集合
    n_train_users = train["UserID"].nunique()
    for name, part in (("test", test), ("valid", valid)):
        counts = part.groupby("UserID").size()
        assert (counts == 1).all(), f"{name} 中存在用户记录数不为 1 的情况"
        assert counts.size == n_train_users, (
            f"{name} 覆盖 {counts.size} 个用户，train 覆盖 {n_train_users} 个，不一致"
        )

    # 3. 时间序单调：train 的最后一条不晚于 valid 的第一条，valid 不晚于 test
    def _edge(part: pd.DataFrame, how: str) -> pd.Series:
        return part.groupby("UserID")["Timestamp"].agg(how)

    train_max, valid_max = _edge(train, "max"), _edge(valid, "max")
    valid_min, test_min = _edge(valid, "min"), _edge(test, "min")

    for earlier_max, later_min, msg in (
        (train_max, valid_min, "存在用户在 train 中有晚于 valid 的记录"),
        (valid_max, test_min, "存在用户在 valid 中有晚于 test 的记录"),
    ):
        users = earlier_max.index.intersection(later_min.index)
        assert (earlier_max.reindex(users) <= later_min.reindex(users)).all(), msg


def coverage_report(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """报告各集合的规模与冷启动情况（unseen = 未在 train 中出现过）。"""
    train = splits["train"]
    train_movies = set(train["MovieID"])
    train_users = set(train["UserID"])

    rows = []
    for name in SPLIT_NAMES:
        part = splits[name]
        is_train = name == "train"
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "users": part["UserID"].nunique(),
                "movies": part["MovieID"].nunique(),
                "unseen_movies": 0 if is_train else len(set(part["MovieID"]) - train_movies),
                "unseen_users": 0 if is_train else len(set(part["UserID"]) - train_users),
            }
        )
    return pd.DataFrame(rows)


def save_splits(splits: dict[str, pd.DataFrame]) -> dict[str, str]:
    """保存为 parquet（outputs/ 已被 git 忽略），返回相对路径。"""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SPLIT_NAMES:
        path = SPLITS_DIR / f"{name}.parquet"
        splits[name].to_parquet(path, index=False)
        paths[name] = str(path.relative_to(config.ROOT))
    return paths


def load_splits() -> dict[str, pd.DataFrame]:
    """读回已保存的划分。"""
    return {name: pd.read_parquet(SPLITS_DIR / f"{name}.parquet") for name in SPLIT_NAMES}


if __name__ == "__main__":
    from .data_loader import load_ratings

    ratings = load_ratings()
    splits = leave_one_out_split(ratings)
    assert_no_leakage(splits)

    report = coverage_report(splits)
    print("== 划分结果 ==")
    print(report.to_string(index=False))

    total = sum(len(p) for p in splits.values())
    print(f"\n合计 {total} 条，与原始 {len(ratings)} 条一致: {total == len(ratings)}")
    print("防泄漏断言全部通过")

    paths = save_splits(splits)
    print("\n== 已保存 ==")
    for name, path in paths.items():
        print(f"  {name:5s} -> {path}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.RESULTS_DIR / "split_summary.txt"
    summary_path.write_text(
        report.to_string(index=False)
        + f"\n\n合计 {total} 条；防泄漏断言通过\n",
        encoding="utf-8",
    )
    print(f"  摘要 -> {summary_path.relative_to(config.ROOT)}")
