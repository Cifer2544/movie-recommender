"""Item-based collaborative filtering（课程算法，见 Lecture 7）。

算法定义
--------
1. 相似度：adjusted cosine，先按用户减均值（抵消用户评分尺度差异），
   再对物品向量做余弦：

       sim(i,j) = Σ_u (r_ui - μ_u)(r_uj - μ_u)
                  / sqrt(Σ_u (r_ui - μ_u)²) · sqrt(Σ_u (r_uj - μ_u)²)

   在 userId-itemId 的中心化稀疏矩阵上等价于 S = CᵀC 再按列范数归一化。

2. 流行度惩罚（可选，默认关闭）：

       sim'(i,j) = sim(i,j) / (|N(i)| · |N(j)|)^(α/2)

   |N(i)| 是训练集中物品 i 的评分数。α=0 时不做任何惩罚（标准 adjusted
   cosine）；α=1 时按几何平均评分数做对称衰减，抑制热门物品支配相似度。

   注意方向：惩罚项必须是"除以评分数"，α 增大 = 更强地压低热门物品。
   若写成 n^(-α) 且不设下界，α 会反向放大冷门物品，导致 top-K 邻居全被
   冷门片占据（实测 Recall@10 从 4% 掉到 1.3%，比流行度基线还差）。

3. 邻居截断：每个物品只保留相似度最高的 K 个邻居（其余置 0），
   降低噪声与计算量。

4. 打分聚合（Top-N 排序用）：

       score(u,j) = Σ_{i∈N(u)} sim'(i,j) · w_ui

   对应讲义 P27 的第 3 步"Normalize ratings and compute a prediction from a
   weighted combination of the selected neighbors' ratings"。w_ui 由
   `weighting` 决定（在验证集上选型，见 tune_and_evaluate）：

       binary    w_ui = 1           纯共现，最接近经典 ItemKNN（验证集最优，默认值）
       rating    w_ui = r_ui        直接用评分
       centered  w_ui = r_ui - μ_u  按用户均值归一化

   为什么用"求和"而不是"加权平均"：加权平均会让每个候选物品的分数都收敛到
   用户均分附近（差异极小），Top-N 排序退化为噪声，实测 Recall@10 只有 1.3%；
   求和保留了"有多少个相似邻居、相似度多强"的信息，才能区分候选物品。
   这也是 ItemKNN 系列实现（如 RecBole 的 `interaction @ sim_matrix`）的通行做法。

防泄漏
------
相似度矩阵、物品评分数 |N(i)|、用户均值 μ_u 全部只由 **train** 计算；
valid / test 不参与 fit，只在打分与评估阶段使用。

扩展性
------
(Cn.T @ Cn).toarray() 是 n_items² 的稠密矩阵：ML-1M（3704 部）约 55MB，可接受；
换到 ML-20M / ML-25M 量级需改为按行分块取 top-K，或全程保持稀疏。
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import sparse

from .. import config
from ..evaluate import TrainIndex, evaluate_scores, format_metrics, report_metrics


class ItemCF:
    """基于物品相似度的邻域推荐模型。"""

    def __init__(
        self,
        alpha: float = 0.0,
        k_neighbors: int = 50,
        weighting: str = "binary",
        drop_negative: bool = False,
    ):
        if weighting not in ("centered", "binary", "rating"):
            raise ValueError(f"weighting 必须是 centered/binary/rating，收到 {weighting}")
        if alpha < 0:
            raise ValueError(f"alpha 必须 >= 0，收到 {alpha}")
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors 必须 >= 1，收到 {k_neighbors}")
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.weighting = weighting
        self.drop_negative = drop_negative

    # -----------------------------------------------------------
    # 训练（此处训练 = 预计算相似度矩阵，无梯度更新）
    # -----------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "ItemCF":
        # csr 构造对重复 (行, 列) 会**求和**评分，|N(i)| 也会被高估，这里先拦下
        if train.duplicated(subset=["UserID", "MovieID"]).any():
            raise ValueError(
                "训练集存在重复的 (UserID, MovieID)：csr 会求和评分、getnnz 会高估 |N(i)|，请先聚合或去重"
            )

        self.users_ = np.sort(train["UserID"].unique())
        self.items_ = np.sort(train["MovieID"].unique())
        self.user_index_ = {u: i for i, u in enumerate(self.users_)}
        self.item_index_ = {m: i for i, m in enumerate(self.items_)}

        rows = train["UserID"].map(self.user_index_).to_numpy()
        cols = train["MovieID"].map(self.item_index_).to_numpy()
        vals = train["Rating"].to_numpy(dtype=np.float32)
        # getnnz 只数非零元素，显式 0 会漏计 |N(i)|
        if (vals == 0).any():
            raise ValueError("训练集评分含 0：getnnz 不计显式 0，会低估 |N(i)|，请先核对数据")

        self.R_ = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.users_), len(self.items_))
        )

        # 每个物品的评分数 |N(i)|，用于流行度惩罚
        item_counts = np.asarray(self.R_.getnnz(axis=0), dtype=np.float64)
        self.item_counts_ = item_counts

        # 按用户中心化：C = R - μ_u（只对已评分位置生效）
        user_counts = np.asarray(self.R_.getnnz(axis=1), dtype=np.float64)
        user_sums = np.asarray(self.R_.sum(axis=1)).ravel()
        user_means = np.divide(
            user_sums, user_counts, out=np.zeros_like(user_sums), where=user_counts > 0
        )

        C = self.R_.copy().astype(np.float32)
        row_means = np.repeat(user_means, np.diff(C.indptr)).astype(np.float32)
        C.data -= row_means
        C.eliminate_zeros()
        self.C_ = C  # 中心化评分矩阵，打分阶段复用
        self.user_means_ = user_means

        # 二值交互矩阵（weighting="binary" 时使用）
        B = self.R_.copy().astype(np.float32)
        B.data = np.ones_like(B.data)
        self.B_ = B

        # adjusted cosine：按列 L2 范数归一化后做 CᵀC
        col_norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=0)).ravel())
        col_norms[col_norms == 0] = 1.0
        inv_norms = sparse.diags(1.0 / col_norms.astype(np.float32))
        Cn = C @ inv_norms

        sim = (Cn.T @ Cn).toarray().astype(np.float32)
        np.fill_diagonal(sim, 0.0)

        # 惩罚必须在截断之前：惩罚会改变行内排序，先截断再惩罚与直接惩罚后截断不等价
        sim = self._apply_popularity(sim, item_counts, self.alpha, inplace=True)
        if self.drop_negative:
            # 正邻居不足 k 个时，负相似度会占掉名额；置 0 后贡献为 0，等于不参与聚合
            sim[sim < 0] = 0.0

        self.sim_ = self._top_k(sim, self.k_neighbors)
        return self

    @staticmethod
    def _apply_popularity(
        sim: np.ndarray, item_counts: np.ndarray, alpha: float, inplace: bool = False
    ) -> np.ndarray:
        """按 (|N(i)|·|N(j)|)^(α/2) 对称衰减相似度；alpha=0 时原样返回。

        inplace=False 时保证不修改入参，便于调参阶段在同一份未截断的 sim 上
        反复施加不同 alpha（只做一次 CᵀC 就能扫完整张 alpha 网格）。
        """
        if alpha <= 0:
            return sim if inplace else sim.copy()
        damp = np.power(np.maximum(item_counts, 1.0), alpha / 2.0).astype(np.float32)
        return sim / np.outer(damp, damp)

    @staticmethod
    def _top_k(sim: np.ndarray, k: int) -> np.ndarray:
        """每行只保留最大的 k 个相似度，其余置 0。"""
        n = sim.shape[0]
        if k >= n:
            return sim
        out = np.zeros_like(sim)
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        np.put_along_axis(out, idx, np.take_along_axis(sim, idx, axis=1), axis=1)
        return out

    # -----------------------------------------------------------
    # 打分与推荐
    # -----------------------------------------------------------
    def _weight_matrix(self) -> sparse.csr_matrix:
        """按 weighting 选择用户侧权重矩阵（centered=中心化评分，binary=是否交互，rating=原始评分）。"""
        return {"centered": self.C_, "binary": self.B_, "rating": self.R_}[self.weighting]

    def score_users(self, users=None) -> tuple[np.ndarray, np.ndarray]:
        """返回 (users, scores)，scores 形状 (n_users, n_items)，两者行严格对应。

        打分 = 用户侧权重矩阵 · Sim'，权重由 `weighting` 决定：
            centered  中心化评分（减用户均值，突出"高于自己平均口味"的部分）
            binary    是否交互过（纯共现计数，最接近经典 ItemKNN）
            rating    原始评分（1-5 直接加权）

        传入的 users 若含训练集未出现的用户，会用同一个 mask 从返回的 users 中
        一并剔除，避免出现"users 有 N 个、scores 只有 M 行"而调用方按下标取
        值时静默错位。
        """
        if users is None:
            users = self.users_
            M = self._weight_matrix()
        else:
            users = np.asarray(users)
            mask = np.fromiter(
                (int(u) in self.user_index_ for u in users), dtype=bool, count=len(users)
            )
            users = users[mask]
            rows = [self.user_index_[int(u)] for u in users]
            M = self._weight_matrix()[rows] if rows else self._weight_matrix()[:0]

        scores = np.asarray(M @ self.sim_)
        return users, scores

    def score_user(self, user_id: int) -> np.ndarray:
        """单个用户的打分向量（长度 = len(items_)）；训练集未见过的用户返回全 0。"""
        users, scores = self.score_users([user_id])
        return scores[0] if len(users) else np.zeros(len(self.items_), dtype=np.float32)

    def top_k(self, user_id: int, k: int = config.TOP_K) -> list[tuple[int, float]]:
        """返回 [(MovieID, score)]，已屏蔽该用户在 train 中评过的电影。"""
        user_id = int(user_id)
        if user_id not in self.user_index_:
            return []

        scores = self.score_user(user_id)
        scores[self.R_[self.user_index_[user_id]].indices] = -np.inf

        # argpartition 的 kth 合法范围是 [0, n-1]：k == len(scores) 时 kth 会越界
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            (int(self.items_[j]), float(scores[j])) for j in top if np.isfinite(scores[j])
        ]


# ---------------------------------------------------------------
# 网格搜索 + 评估（python -m src.models.itemcf）
# ---------------------------------------------------------------
def tune_and_evaluate():
    from ..data_loader import load_all
    from ..split import load_splits, leave_one_out_split

    # 只加载一次：构造 splits 的兜底与结尾查片名共用，避免全量读两遍
    ratings, movies, _ = load_all()

    try:
        splits = load_splits()
    except FileNotFoundError:
        splits = leave_one_out_split(ratings)

    weightings = ("centered", "binary", "rating")
    alphas = (0.0, 0.25, 0.5, 1.0)
    neighbor_grid = (20, 50, 100, 200)
    max_k = max(neighbor_grid)

    # evaluate_scores 统一读取 splits["test"]，这里用别名把验证集当成 test 传入，
    # 保证超参选择只看验证集，不碰测试集。
    valid_splits = {"train": splits["train"], "test": splits["valid"]}
    test_splits = {"train": splits["train"], "test": splits["test"]}

    # 相似度只依赖数据，与 alpha / weighting / k 都无关：全量 CᵀC 只算一次。
    # k_neighbors=n_items 让 fit 跳过截断，得到未惩罚、未截断的 sim_raw。
    n_items = splits["train"]["MovieID"].nunique()
    base = ItemCF(alpha=0.0, k_neighbors=n_items, weighting="binary").fit(splits["train"])
    sim_raw = base.sim_
    # 训练集索引只依赖 (train, items)，48 次评估共用一份，避免反复 groupby 98 万行
    train_index = TrainIndex(splits["train"], base.items_)

    rows = []
    best = None
    for alpha in alphas:
        # 顺序不能反：惩罚会改变行内排序，先截断再惩罚会得到不同的邻居集合
        sim_alpha = ItemCF._apply_popularity(
            sim_raw, base.item_counts_, alpha, inplace=False
        )
        sim_alpha = ItemCF._top_k(sim_alpha, max_k)

        for weighting in weightings:
            model = copy.copy(base)
            model.alpha = alpha
            model.weighting = weighting
            for k in neighbor_grid:
                model.sim_ = ItemCF._top_k(sim_alpha, k)
                users, scores = model.score_users()
                m = evaluate_scores(
                    scores,
                    users,
                    model.items_,
                    valid_splits,
                    relevant_only=False,
                    train_index=train_index,
                )
                rows.append(
                    {
                        "weighting": weighting,
                        "alpha": alpha,
                        "k": k,
                        "recall@10": round(m["recall@10"], 5),
                        "ndcg@10": round(m["ndcg@10"], 5),
                        "mrr@10": round(m["mrr@10"], 5),
                    }
                )
                if best is None or m["ndcg@10"] > best[0]:
                    best = (m["ndcg@10"], weighting, alpha, k)

    table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    print("== 验证集网格搜索（指标越高越好） ==")
    print(table.to_string(index=False))

    _, best_weighting, best_alpha, best_k = best
    print(f"\n最佳配置: weighting={best_weighting}, alpha={best_alpha}, k_neighbors={best_k}")

    # 用最佳配置评估（同样从 sim_raw 派生，不重新 fit）
    model = copy.copy(base)
    model.alpha = best_alpha
    model.weighting = best_weighting
    sim_best = ItemCF._apply_popularity(
        sim_raw, base.item_counts_, best_alpha, inplace=False
    )
    model.sim_ = ItemCF._top_k(sim_best, best_k)

    report_metrics(model, valid_splits, train_index, "验证集（同一配置）")
    report_metrics(model, test_splits, train_index, "测试集（最终报告口径）")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "itemcf_tuning.txt"
    out.write_text(
        table.to_string(index=False)
        + f"\n\n最佳: weighting={best_weighting}, alpha={best_alpha}, k={best_k}\n",
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out.relative_to(config.ROOT)}")

    # ---- 负相似度置零是否有效：只在最佳配置上做一次 A/B，避免把网格翻倍 ----
    print("\n== 负相似度置零 A/B（最佳配置，验证集）==")
    for drop in (False, True):
        sim_ab = ItemCF._apply_popularity(
            sim_raw, base.item_counts_, best_alpha, inplace=False
        )
        if drop:
            sim_ab[sim_ab < 0] = 0.0
        probe = copy.copy(model)
        probe.drop_negative = drop
        probe.sim_ = ItemCF._top_k(sim_ab, best_k)
        users, scores = probe.score_users()
        r = evaluate_scores(
            scores, users, probe.items_, valid_splits, relevant_only=False, train_index=train_index
        )
        print(f"  drop_negative={str(drop):5s}  {format_metrics(r, show_excl_cold=False)}")

    # ---- 最终模型口径②：train+valid 重训后再评 test（标准 leave-one-out 做法）----
    # 上方那组是"纯 train 训练、valid 只调参、不重训"，两组并列即可证明
    # valid 并入与否对结论无影响。重训后已交互屏蔽也必须用 train+valid。
    full_train = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    full_splits = {"train": full_train, "test": splits["test"]}
    final = ItemCF(
        alpha=best_alpha, k_neighbors=best_k, weighting=best_weighting
    ).fit(full_train)
    # 重训后物品表可能变大（valid 里有 train 没出现过的电影），索引要重建
    full_index = TrainIndex(full_train, final.items_)

    report_metrics(final, full_splits, full_index, "测试集（train+valid 重训）")

    # 示例推荐用口径②的 final（最终交付模型，见过 train+valid，屏蔽集同样是 full_train）
    movie_title = movies.set_index("MovieID")[["Title", "Genres"]]
    print("\n== 示例推荐：UserID=1（Top 5，train+valid 重训模型，已交互已屏蔽）==")
    for movie_id, score in final.top_k(1, 5):
        title, genres = movie_title.loc[movie_id]
        print(f"  {score:7.3f}  {movie_id:4d}  {title}  [{genres}]")

    return model, table, (best_weighting, best_alpha, best_k)


if __name__ == "__main__":
    tune_and_evaluate()
