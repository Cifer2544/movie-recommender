"""BPR（Bayesian Personalized Ranking, Rendle et al., UAI 2009）的 PyTorch CPU 实现。

算法假设
--------
BPR 优化 pairwise 偏序：对用户 u，观测到的正反馈物品 i 应排在未观测物品 j 之前，
最大化 P(i >_u j) = σ(<u,i> − <u,j>) 的后验，等价于最小化
−log σ(x_uij) + L2 正则，直接对齐 Top-N 排序目标，而不是回归评分数值。

为什么用 binary 而非评分
------------------------
Top-N 只关心"喜欢 vs 未观测"的偏序；评分绝对值带用户尺度噪声，低分也不等于
"不喜欢"。故按 preprocess.py 的标签定义，把 train 中 Rating >=
config.POSITIVE_RATING_THRESHOLD 的交互二元化为正例，不使用评分值。

负采样池
--------
对用户 u：候选负例 = items_ 全集 − u 在 train 中评分 >= 4 的电影。
u 评过 1-3 分的电影仍可作负例（"未观测"，不是"明确不喜欢"）。
每个正例采 1 个负例（与原论文一致），拒绝采样实现均匀抽取。

防泄漏（逐条对应实现位置）
--------------------------
1. 负采样只从 train 内部采：fit() 用 train 正例构建 self._pos_mask，
   _sample_negatives() 只访问该掩码与 items_ 全集，全程不接触 valid/test。
2. 用户/物品索引、正例列表、负例池全部只由 train 构建。
3. 早停看 valid 不看 test：fit(train, valid=...) 的早停分支只评估传入的 valid；
   tune_and_evaluate() 网格只传 splits["valid"]，test 仅在最佳组合上评一次。
4. 早停评估复用 self.train_index_（只由 train 构建），不因评估而接触其他集合。

评估口径
--------
score_users() 返回未屏蔽的原始打分（与 ItemCF 完全一致），train 交互的屏蔽由
evaluate_scores 统一负责；top_k() 内部屏蔽该用户在 train 中全部已交互电影。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F

from .. import config
from ..evaluate import TrainIndex, evaluate_scores, report_metrics


class BPR:
    """矩阵分解式 BPR：pairwise 损失 + Adam，早停保留最佳 epoch 参数。"""

    _EVAL_EVERY: int = 2  # 每 2 个 epoch 在 valid 上评一次（最后一轮必评）
    _MAX_RESAMPLE: int = 100  # 负采样重采上限，仅用于防御

    def __init__(
        self,
        embed_dim: int = 64,
        lr: float = 1e-3,
        reg: float = 1e-5,
        batch_size: int = 4096,
        max_epochs: int = 50,
        patience: int = 3,
        seed: int = config.SEED,
    ) -> None:
        if embed_dim < 1 or batch_size < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("embed_dim/batch_size/max_epochs/patience 必须 >= 1")
        if lr <= 0 or reg < 0:
            raise ValueError("lr 必须 > 0，reg 必须 >= 0")
        self.embed_dim = embed_dim
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    # -----------------------------------------------------------
    # 训练
    # -----------------------------------------------------------
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "BPR":
        """训练模型。

        valid 为可选早停集（fit(train) 的调用形式不变；不传则训满 max_epochs）。
        早停只读 valid，绝不读 test。
        """
        if train.duplicated(subset=["UserID", "MovieID"]).any():
            raise ValueError("训练集存在重复的 (UserID, MovieID)，请先聚合或去重")

        # ---- 索引：保留 train 中出现的全部用户/物品，不因正反馈过滤而缩减 ----
        self.users_ = np.sort(train["UserID"].unique())
        self.items_ = np.sort(train["MovieID"].unique())
        self.user_index_ = {u: i for i, u in enumerate(self.users_)}
        self.item_index_ = {m: i for i, m in enumerate(self.items_)}
        n_users, n_items = len(self.users_), len(self.items_)

        rows = train["UserID"].map(self.user_index_).to_numpy(dtype=np.int64)
        cols = train["MovieID"].map(self.item_index_).to_numpy(dtype=np.int64)
        ratings = train["Rating"].to_numpy()

        # top_k 屏蔽用：train 全部交互（任意评分），与 ItemCF 口径一致
        self._rated = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )

        # ---- 正例：Rating >= 阈值，二元化，按 (u, i) 去重，只来自 train ----
        pos = ratings >= config.POSITIVE_RATING_THRESHOLD
        pairs = np.unique(rows[pos] * n_items + cols[pos])
        pu, pi = pairs // n_items, pairs % n_items

        # 先剔除"正例覆盖全集"的用户（无负例可采），再建掩码。
        # 顺序反了会让这些用户永远采不到负例，拒绝采样陷入死循环。
        n_pos_per_user = np.bincount(pu, minlength=n_users)
        keep = n_pos_per_user[pu] < n_items
        pu, pi = pu[keep], pi[keep]
        if len(pu) == 0:
            raise ValueError(
                f"训练集没有评分 >= {config.POSITIVE_RATING_THRESHOLD} 的正反馈，无法训练 BPR"
            )

        self._pos_mask = np.zeros((n_users, n_items), dtype=bool)
        self._pos_mask[pu, pi] = True
        n_pos = len(pu)

        # 早停与外部评估共用：训练集索引只建一次，不必每轮 groupby 98 万行
        self.train_index_ = TrainIndex(train, self.items_)
        valid_splits = {"train": train, "test": valid} if valid is not None else None

        # ---- 模型：CPU，normal(std=0.01)，固定 seed 可复现 ----
        torch.manual_seed(self.seed)
        self.user_emb = nn.Embedding(n_users, self.embed_dim)
        self.item_emb = nn.Embedding(n_items, self.embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        optimizer = torch.optim.Adam(
            list(self.user_emb.parameters()) + list(self.item_emb.parameters()),
            lr=self.lr,
        )
        rng = np.random.default_rng(self.seed)

        best_ndcg = -np.inf
        best_state: tuple[torch.Tensor, torch.Tensor] | None = None
        bad_rounds = 0
        self.best_epoch_ = 0

        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(n_pos)  # 每 epoch 打乱正例顺序
            for start in range(0, n_pos, self.batch_size):
                idx = order[start : start + self.batch_size]
                u, i = pu[idx], pi[idx]
                j = self._sample_negatives(u, rng)
                vu = self.user_emb(torch.from_numpy(u))
                vi = self.item_emb(torch.from_numpy(i))
                vj = self.item_emb(torch.from_numpy(j))
                x_uij = (vu * vi).sum(dim=1) - (vu * vj).sum(dim=1)
                # 逐 triplet 公式 −logσ(x) + reg*(||vi||²+||vj||²)，batch 取均值；
                # 按题设只正则物品向量，不含用户项
                loss = -F.logsigmoid(x_uij).mean() + self.reg * (
                    vi.pow(2).sum(dim=1) + vj.pow(2).sum(dim=1)
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ---- 早停：只看 valid，每 2 个 epoch 一次，最后一轮必评 ----
            if valid_splits is not None and (
                epoch % self._EVAL_EVERY == 0 or epoch == self.max_epochs
            ):
                users, scores = self.score_users()
                m = evaluate_scores(
                    scores,
                    users,
                    self.items_,
                    valid_splits,
                    relevant_only=False,
                    train_index=self.train_index_,
                )
                if m["ndcg@10"] > best_ndcg:
                    best_ndcg = m["ndcg@10"]
                    self.best_epoch_ = epoch
                    best_state = (
                        self.user_emb.weight.detach().clone(),
                        self.item_emb.weight.detach().clone(),
                    )
                    bad_rounds = 0
                else:
                    bad_rounds += 1  # patience 按"评估轮"计数
                    if bad_rounds >= self.patience:
                        break

        if best_state is not None:  # 回滚到最佳 epoch 参数
            self.user_emb.weight.data.copy_(best_state[0])
            self.item_emb.weight.data.copy_(best_state[1])
        else:
            self.best_epoch_ = self.max_epochs
        return self

    def _sample_negatives(self, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """每个正例均匀采 1 个负例：items_ 全集 − 该用户 train 正例。

        拒绝采样：全集均匀采，撞上正例则重采（正例占比不到 2%，几乎一次命中）。
        只访问 train 构建的 _pos_mask，不接触 valid/test。
        """
        n_items = self._pos_mask.shape[1]
        j = rng.integers(0, n_items, size=len(u))
        hit = self._pos_mask[u, j]
        tries = 0
        while hit.any():
            tries += 1
            if tries > self._MAX_RESAMPLE:
                # 不可达：正例覆盖全集的用户在 fit() 里已被剔除
                raise RuntimeError("负采样反复冲突，请检查 _pos_mask 的构建顺序")
            j[hit] = rng.integers(0, n_items, size=int(hit.sum()))
            hit = self._pos_mask[u, j]
        return j

    # -----------------------------------------------------------
    # 打分与推荐
    # -----------------------------------------------------------
    @torch.no_grad()
    def score_users(self, users=None) -> tuple[np.ndarray, np.ndarray]:
        """(users, scores)：scores = U·Vᵀ，dense float32，行与返回的 users 对齐。

        与 ItemCF 一致：不在此屏蔽 train 交互，屏蔽统一交给 evaluate_scores；
        传入未知用户时同步过滤，避免调用方按下标取值时静默错位。
        """
        U = self.user_emb.weight.detach().numpy()
        V = self.item_emb.weight.detach().numpy()
        if users is None:
            users = self.users_
            scores = U @ V.T
        else:
            users = np.asarray(users)
            keep = np.fromiter(
                (int(u) in self.user_index_ for u in users), dtype=bool, count=len(users)
            )
            users = users[keep]
            rows = np.array([self.user_index_[int(u)] for u in users], dtype=np.int64)
            scores = U[rows] @ V.T if len(rows) else U[:0] @ V.T
        return users, np.asarray(scores, dtype=np.float32)

    @torch.no_grad()
    def score_user(self, user_id: int) -> np.ndarray:
        """单用户打分向量（长度 = len(items_)）；未知用户返回全 0。"""
        user_id = int(user_id)
        if user_id not in self.user_index_:
            return np.zeros(len(self.items_), dtype=np.float32)
        u = self.user_emb.weight[self.user_index_[user_id]]
        return (u @ self.item_emb.weight.T).numpy().astype(np.float32)

    def top_k(self, user_id: int, k: int = config.TOP_K) -> list[tuple[int, float]]:
        """返回 [(MovieID, score)]，已屏蔽该用户在 train 中评过的电影。"""
        user_id = int(user_id)
        if user_id not in self.user_index_:
            return []

        scores = self.score_user(user_id)
        scores[self._rated[self.user_index_[user_id]].indices] = -np.inf

        k = min(k, len(scores))
        if k <= 0:
            return []
        # argpartition 的 kth 合法范围是 [0, n-1]：k == len(scores) 时 kth 会越界
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            (int(self.items_[j]), float(scores[j])) for j in top if np.isfinite(scores[j])
        ]

    # -----------------------------------------------------------
    # 缓存接口（见 src/cache.py）
    # -----------------------------------------------------------
    def hparams(self) -> dict:
        """影响训练结果的全部构造参数，seed 必须在内（否则不同 seed 会撞同一缓存）。"""
        return {
            "embed_dim": self.embed_dim,
            "lr": self.lr,
            "reg": self.reg,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
        }

    def get_state(self) -> dict:
        """只存推理所需：两张嵌入表 + 屏蔽用的 _rated。

        训练期的 _pos_mask（(n_users, n_items) bool，约 22 MB）与 train_index_ 不存。
        """
        return {
            "users": self.users_,
            "items": self.items_,
            "user_emb": self.user_emb.weight.detach().numpy(),
            "item_emb": self.item_emb.weight.detach().numpy(),
            "rated": self._rated,
        }

    def set_state(self, state: dict, meta: dict) -> "BPR":
        self.users_ = np.asarray(state["users"])
        self.items_ = np.asarray(state["items"])
        self.user_index_ = {u: i for i, u in enumerate(self.users_)}
        self.item_index_ = {m: i for i, m in enumerate(self.items_)}

        n_users, n_items = len(self.users_), len(self.items_)
        self.user_emb = nn.Embedding(n_users, self.embed_dim)
        self.item_emb = nn.Embedding(n_items, self.embed_dim)
        with torch.no_grad():
            self.user_emb.weight.copy_(torch.from_numpy(state["user_emb"]))
            self.item_emb.weight.copy_(torch.from_numpy(state["item_emb"]))

        self._rated = state["rated"]
        self._pos_mask = None  # 仅负采样阶段使用，推理不需要
        self.train_index_ = None  # evaluate_scores 收到 None 时会自行构建
        self.best_epoch_ = int(meta.get("best_epoch", 0))
        return self


# ---------------------------------------------------------------
# 网格搜索 + 评估（python -m src.models.bpr）
# ---------------------------------------------------------------
def tune_and_evaluate() -> tuple[BPR, pd.DataFrame, tuple[int, float, float]]:
    """8 组超参只在 valid 上选；test 仅对最佳组合评一次。"""
    from ..data_loader import load_all
    from ..split import load_splits, leave_one_out_split

    # 只加载一次：兜底划分与结尾查片名共用
    ratings, movies, _ = load_all()

    try:
        splits = load_splits()
    except FileNotFoundError:
        splits = leave_one_out_split(ratings)

    valid_splits = {"train": splits["train"], "test": splits["valid"]}
    test_splits = {"train": splits["train"], "test": splits["test"]}

    grid = [
        (embed_dim, lr, reg)
        for embed_dim in (32, 64)
        for lr in (1e-3, 5e-3)
        for reg in (1e-5, 1e-4)
    ]

    rows: list[dict] = []
    best = None
    best_model: BPR | None = None
    for n, (embed_dim, lr, reg) in enumerate(grid, 1):
        model = BPR(embed_dim=embed_dim, lr=lr, reg=reg).fit(
            splits["train"], valid=splits["valid"]
        )
        users, scores = model.score_users()
        m = evaluate_scores(
            scores,
            users,
            model.items_,
            valid_splits,
            relevant_only=False,
            train_index=model.train_index_,
        )
        print(
            f"  [{n}/{len(grid)}] embed_dim={embed_dim} lr={lr:g} reg={reg:g} "
            f"best_epoch={model.best_epoch_:2d}  NDCG@10={m['ndcg@10']:.4f}",
            flush=True,
        )
        rows.append(
            {
                "embed_dim": embed_dim,
                "lr": lr,
                "reg": reg,
                "best_epoch": model.best_epoch_,
                "recall@10": round(m["recall@10"], 5),
                "ndcg@10": round(m["ndcg@10"], 5),
                "mrr@10": round(m["mrr@10"], 5),
            }
        )
        if best is None or m["ndcg@10"] > best[0]:
            best = (m["ndcg@10"], embed_dim, lr, reg)
            best_model = model  # 固定 seed，复用等价于重训，省一次完整训练

    table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    print("\n== 验证集网格搜索（指标越高越好） ==")
    print(table.to_string(index=False))

    _, best_dim, best_lr, best_reg = best
    model = best_model
    print(f"\n最佳配置: embed_dim={best_dim}, lr={best_lr}, reg={best_reg}")

    report_metrics(model, valid_splits, model.train_index_, "验证集（同一配置）")
    report_metrics(model, test_splits, model.train_index_, "测试集（最终报告口径）")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "bpr_tuning.txt"
    out.write_text(
        table.to_string(index=False)
        + f"\n\n最佳: embed_dim={best_dim}, lr={best_lr}, reg={best_reg}\n",
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out.relative_to(config.ROOT)}")

    # ---- 最终模型口径②：train+valid 重训后再评 test（与 ItemCF 并列）----
    # 没有独立 valid 可早停，只能固定按最佳模型的 best_epoch 轮数训练。
    # 这是 leave-one-out 重训的固有局限：数据只多 0.6%，但轮数不能随数据自适应，
    # 若最佳模型恰好在很早期收敛，重训即欠拟合。本数据实测重训后指标更高
    # （0.0632 → 0.0679），风险未显化；该口径仅作参考，主要结论以 train 行为准。
    full_train = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    final = BPR(
        embed_dim=best_dim, lr=best_lr, reg=best_reg, max_epochs=model.best_epoch_
    ).fit(full_train)
    # 重训后物品表可能变大（valid 里有 train 没出现过的电影），掩码索引由 fit 按 full_train 重建
    report_metrics(
        final,
        {"train": full_train, "test": splits["test"]},
        final.train_index_,
        "测试集（train+valid 重训）",
    )

    # 示例推荐用口径②的 final：它是最终交付模型（见过 train+valid），
    # 屏蔽集也来自 full_train，与 Web 端推理时的实际行为一致
    movie_title = movies.set_index("MovieID")[["Title", "Genres"]]
    print("\n== 示例推荐：UserID=1（Top 5，train+valid 重训模型，已交互已屏蔽）==")
    for movie_id, score in final.top_k(1, 5):
        title, genres = movie_title.loc[movie_id]
        print(f"  {score:7.3f}  {movie_id:4d}  {title}  [{genres}]")

    return model, table, (best_dim, best_lr, best_reg)


if __name__ == "__main__":
    tune_and_evaluate()
