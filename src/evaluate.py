"""统一评估：leave-one-out 口径下的排序指标。

由于本项目的测试集是"每用户 1 条留出交互"，Recall@K 与 HitRate@K 在数学上
完全等价（都是"该用户的留出电影是否进入 Top-K"的比例），因此只报告
Recall@K、NDCG@K、MRR@K 三个指标，不再重复报 HitRate。

相关性口径与 src/preprocess.py 保持一致：Rating >= 4 才算 relevant。
`relevant_only=True` 时只评测留出评分为正反馈的用户（约 58%），
`False` 时把留出电影一律视为相关（便于与序列推荐论文的常规口径对比）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _train_items_by_user(train: pd.DataFrame, item_col: pd.Series) -> dict[int, np.ndarray]:
    cols = item_col.reindex(train["MovieID"]).to_numpy()
    frame = pd.DataFrame({"UserID": train["UserID"].to_numpy(), "col": cols})
    frame = frame.dropna(subset=["col"])
    frame["col"] = frame["col"].astype("int32")
    return {
        int(uid): grp["col"].to_numpy()
        for uid, grp in frame.groupby("UserID", sort=False)
    }


class TrainIndex:
    """训练集索引的预计算缓存：user → 该用户已交互物品的列号。

    网格搜索会调用 evaluate_scores 数十次，每次重新 groupby 全量 train（98 万行）
    是评估环节最大的固定开销；只要训练集与物品表不变，就应当复用同一份。
    """

    __slots__ = ("items", "item_col", "by_user")

    def __init__(self, train: pd.DataFrame, items: np.ndarray):
        self.items = np.asarray(items)
        self.item_col = pd.Series(
            np.arange(len(self.items), dtype="int32"), index=self.items
        )
        self.by_user = _train_items_by_user(train, self.item_col)


def evaluate_scores(
    scores: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    splits: dict[str, pd.DataFrame],
    k: int = config.TOP_K,
    relevant_only: bool = True,
    threshold: int = config.POSITIVE_RATING_THRESHOLD,
    train_index: TrainIndex | None = None,
) -> dict:
    """对 `scores`（行=users，列=items）计算 Top-K 排序指标。

    训练集中用户已交互的电影会被屏蔽（置 -inf），保证推荐的"未看过"前提。
    留出电影若不在 items 中（物品冷启动），记为未命中并单独计数。

    `train_index` 可传入预计算的 TrainIndex，避免网格搜索时反复 groupby 全量 train。
    """
    # 统一转 float32：逐用户复制整行 + argpartition 都更快；分数范围 0~10，
    # float32 的 7 位有效数字足够排序，不会改变名次。
    scores = np.asarray(scores, dtype=np.float32)
    if train_index is None:
        train_index = TrainIndex(splits["train"], items)
    elif not np.array_equal(train_index.items, np.asarray(items)):
        raise ValueError("train_index 的物品表与 items 不一致：请按同一模型重建 TrainIndex")
    item_col = train_index.item_col
    train_by_user = train_index.by_user

    test = splits["test"].set_index("UserID")
    test_movie = test["MovieID"].to_dict()
    test_rating = test["Rating"].to_dict()

    # 指标只需累加和，不必逐用户保留列表
    n_eval = n_skipped_low_rating = n_cold_item = hits = 0
    ndcg_sum = mrr_sum = 0.0

    for row, user in enumerate(users):
        user = int(user)
        movie = test_movie.get(user)
        if movie is None:
            continue
        if relevant_only and test_rating[user] < threshold:
            n_skipped_low_rating += 1
            continue
        if movie not in item_col.index:
            # 物品冷启动：计入整体分母并记 0 分（保守口径），
            # 同时单独维护"排除冷启动"的一组指标供报告并列展示
            n_cold_item += 1
            n_eval += 1
            continue

        # 逐用户复制一行用于就地屏蔽；scores 已在上方统一为 float32
        s = scores[row].copy()
        masked = train_by_user.get(user)
        if masked is not None:
            s[masked] = -np.inf
        s[~np.isfinite(s)] = -np.inf

        target = int(item_col[movie])
        if not np.isfinite(s[target]):
            # 候选集被全部屏蔽（理论上不会发生），按未命中处理
            n_eval += 1
            continue

        n_eval += 1
        # argpartition 的 kth 合法范围是 [0, n-1]：候选集小于 k 时 kth 会越界
        kth = min(k, len(s)) - 1
        top = np.argpartition(-s, kth)[:k]
        if target in top:
            hits += 1
            rank = int((s[top] > s[target]).sum()) + 1
            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / np.log2(rank + 1)

    n_excl_cold = n_eval - n_cold_item

    return {
        "k": k,
        "n_eval": n_eval,
        "n_eval_excl_cold": n_excl_cold,
        "skipped_low_rating": n_skipped_low_rating,
        "cold_items": n_cold_item,
        f"recall@{k}": hits / n_eval if n_eval else float("nan"),
        f"ndcg@{k}": ndcg_sum / n_eval if n_eval else float("nan"),
        f"mrr@{k}": mrr_sum / n_eval if n_eval else float("nan"),
        f"recall@{k}_excl_cold": hits / n_excl_cold if n_excl_cold else float("nan"),
        f"ndcg@{k}_excl_cold": ndcg_sum / n_excl_cold if n_excl_cold else float("nan"),
        f"mrr@{k}_excl_cold": mrr_sum / n_excl_cold if n_excl_cold else float("nan"),
    }


def format_metrics(metrics: dict, show_excl_cold: bool = True) -> str:
    k = metrics["k"]
    text = (
        f"Recall@{k}={metrics[f'recall@{k}']:.4f}  "
        f"NDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
        f"MRR@{k}={metrics[f'mrr@{k}']:.4f}  "
        f"(n={metrics['n_eval']}, cold_items={metrics['cold_items']})"
    )
    if show_excl_cold and metrics["cold_items"]:
        text += (
            f"\n      排除冷启动: Recall@{k}={metrics[f'recall@{k}_excl_cold']:.4f}  "
            f"NDCG@{k}={metrics[f'ndcg@{k}_excl_cold']:.4f}  "
            f"(n={metrics['n_eval_excl_cold']})"
        )
    return text
