"""流行度基线（Most Popular）。

不做任何个性化：所有用户都按训练集上的评分次数排序推荐。
用途有两个：
    1. 作为下界参照——任何个性化模型都应该显著优于它；
    2. 调试评估流程——若基线指标也接近 0，说明问题在评估而不在模型。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MostPopular:
    """按训练集评分数排序的基线推荐器。"""

    def fit(self, train: pd.DataFrame) -> "MostPopular":
        counts = train.groupby("MovieID").size().sort_values(ascending=False)
        self.items_ = counts.index.to_numpy()
        self.counts_ = counts.to_numpy()
        self.users_ = np.sort(train["UserID"].unique())
        return self

    def score_users(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (users, scores)，每行都是同一份流行度分数。"""
        row = self.counts_.astype(np.float32)
        scores = np.tile(row, (len(self.users_), 1))
        return self.users_, scores


if __name__ == "__main__":
    from ..data_loader import load_all
    from ..evaluate import evaluate_scores, format_metrics
    from ..split import load_splits, leave_one_out_split

    try:
        splits = load_splits()
    except FileNotFoundError:
        splits = leave_one_out_split(load_all()[0])

    model = MostPopular().fit(splits["train"])
    users, scores = model.score_users()

    for name in ("valid", "test"):
        one = {"train": splits["train"], "test": splits[name]}
        m = evaluate_scores(scores, users, model.items_, one, relevant_only=False)
        print(f"[{name}] 流行度基线 {format_metrics(m)}")
