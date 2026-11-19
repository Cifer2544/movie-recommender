"""LRURec（Linear Recurrent Units for Sequential Recommendation, Wang et al., WSDM 2024）
的 PyTorch CPU 复现（教学版）。

算法假设
--------
序列推荐：用户下一次交互只依赖其历史正反馈的有序序列。LRURec 用线性循环单元
（LRU）编码序列：h_k = Λ·h_{k-1} + γ⊙B·x_k，y_k = Re(C·h_k) + D⊙x_k，
Λ 为复数对角阵，指数参数化 Λ = exp(-exp(ν) + i·exp(θ)) 保证 |λ|<1（递推稳定）。
相比自注意力，复杂度随序列长度线性增长，且天然携带顺序信息（无需位置嵌入）。

训练目标
--------
many-to-one next-item 预测：给定用户正反馈序列的前 k-1 项，预测第 k 项，
损失为全量 softmax 交叉熵（原论文口径，无负采样）。

注意：只在序列**最后一个位置**计算输出头（即 x[:, -1] @ Eᵀ），而不是对全部
L 个位置都算 logits。后者是 many-to-many，logits 形状 (B, L, n_items)——
128×50×3704 float32 约 95MB，CPU 上每 epoch 要多花约 50 倍时间，且与原论文
的 next-item 目标不一致。

为什么用 binary 而非评分
------------------------
与 BPR/preprocess.py 标签定义一致：Rating >= config.POSITIVE_RATING_THRESHOLD
的交互视为正反馈进入序列，不使用评分值；低分交互不进入序列（视作"未观测"）。

与论文的教学简化
----------------
1. 递推用逐步循环实现（论文用分块并行扫描加速），数学完全等价；
2. 序列截断为最近 max_len 个正例；输出头与物品嵌入权重绑定；
3. d_state 固定等于 d_model（论文中两者独立可调）；
4. PAD 占嵌入第 n_items 行（padding_idx，恒为零），不属于候选集，
   items_/users_ 仍为 train 全集，与 ItemCF/BPR 硬口径一致；
5. 训练目标是 one-target-per-user：只对每个用户序列的末位做一次预测，而论文是
   autoregressive many-to-many（对每个非 PAD 位置都监督下一个 token）。代价是
   中间位置的转移信号全部丢弃。一个用户无论有 57 条还是 1434 条正例都只贡献
   1 个样本（n_train ≈ 6040，比 BPR 的 98 万三元组少两个数量级），训练信号稀疏，
   这是 LRURec 在本实验中表现受限的主要原因之一。
   为什么不保留 many-to-many：输出头 (B, L, d) @ (d, n_items) 在 B=128、L=50、
   n_items=3704 时是 1.5 GFLOP/批，而两个 FFN 块合计只有 0.1 GFLOP/批，
   即输出头占前向计算的 93%；改成末位后是 30 MFLOP/批。这不是"多 30-40%"
   而是约 14 倍，CPU 上 8.5 秒/epoch 会变成 2 分钟/epoch。

防泄漏（逐条对应实现位置）
--------------------------
1. 训练数据只来自 train：fit() 开头由 train 构建索引、_rated、正例序列 _seq
   与训练对 (train_in, train_tg)；全量 softmax 无负采样，不存在采样泄漏。
2. 用户/物品索引、序列全部只由 train 构建，valid/test 交互不进入 fit 的数据侧。
3. 早停看 valid 不看 test：fit(train, valid=...) 早停分支只评估传入的 valid；
   tune_and_evaluate() 网格只传 splits["valid"]，test 仅在最佳组合上评一次。
4. 早停评估复用 self.train_index_（只由 train 构建）。

评估口径
--------
score_users() 用各用户 train 序列最后位置的 next-item logits 作为全物品打分，
不在此屏蔽 train 交互（与 ItemCF/BPR 一致，统一交给 evaluate_scores）；
top_k() 内部屏蔽该用户 train 中全部已交互电影。

冷启动限制（需在报告中说明）
----------------------------
打分完全依赖用户的历史正反馈序列。train 中没有任何正反馈的用户 _seq 全为 PAD，
输出恒为常数向量，评分等同噪声。这是序列推荐模型的固有限制而非实现缺陷：
LRURec 无法为无历史的用户给出有效推荐，ItemCF/BPR 在这类用户上同样退化，
但它们不依赖序列顺序，退化形式不同。
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


class _LRUBlock(nn.Module):
    """单层 LRU 块：LN → 复数线性递推 → 残差 → LN → FFN → 残差。"""

    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # |λ| 均匀初始化在 [0.5, 0.99] 环带，相位均匀 (0, π]
        r = torch.rand(d_state) * (0.99 - 0.5) + 0.5
        theta = torch.rand(d_state) * (np.pi - 1e-3) + 1e-3
        self.nu_log = nn.Parameter(torch.log(-torch.log(r)))
        self.theta_log = nn.Parameter(torch.log(theta))
        self.gamma_log = nn.Parameter(0.5 * torch.log(1.0 - r * r))  # 离散化幅度补偿
        self.B_re = nn.Parameter(torch.randn(d_model, d_state) * (2 * d_model) ** -0.5)
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) * (2 * d_model) ** -0.5)
        self.C_re = nn.Parameter(torch.randn(d_state, d_model) * (2 * d_state) ** -0.5)
        self.C_im = nn.Parameter(torch.randn(d_state, d_model) * (2 * d_state) ** -0.5)
        self.D = nn.Parameter(torch.ones(d_model))  # 跳连（实数对角）
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.ln1(x)  # (B, L, d)
        lam = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
        gamma = torch.exp(self.gamma_log)  # (d_state,)
        B = torch.complex(self.B_re, self.B_im)
        C = torch.complex(self.C_re, self.C_im)
        Bz = z.to(torch.complex64) @ B  # 输入投影一次算完
        if pad_mask is not None:
            # 左填充的 PAD 位置不注入递推状态。padding_idx 只保证嵌入为零，
            # 而 LN(0) = β 是可学习偏置，训练后 β≠0：若照常累加，
            # h 会从 β 漂移起步，经 1/(1-|λ|) 放大后淹没短序列用户的真实信号。
            # pad_mask 为 (B, L, 1)，按末维广播到 Bz (B, L, d_state)
            Bz = Bz * pad_mask
        h = torch.zeros(
            z.shape[0], B.shape[1], dtype=torch.complex64, device=z.device
        )
        hs: list[torch.Tensor] = []
        for t in range(z.shape[1]):  # 逐步递推
            h = lam * h + gamma * Bz[:, t]
            hs.append(h)
        H = torch.stack(hs, dim=1)  # (B, L, d_state)
        y = (H @ C).real + z * self.D
        x = x + self.drop(y)
        x = x + self.ffn(self.ln2(x))
        return x


class _LRUNet(nn.Module):
    """嵌入 → N 个 LRU 块 → LN → 与物品嵌入绑定的输出头。"""

    def __init__(
        self, n_items: int, d_model: int, d_state: int, n_blocks: int, dropout: float
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.pad = n_items  # PAD 行在候选集之外
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=self.pad)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[self.pad].zero_()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_LRUBlock(d_model, d_state, dropout) for _ in range(n_blocks)]
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, seq: torch.Tensor, last_only: bool = True) -> torch.Tensor:
        """seq (B, L) → logits。

        last_only=True（训练与评估都用）只对序列最后一个位置过输出头，
        返回 (B, n_items)；False 时对全部位置计算，返回 (B, L, n_items)，
        仅用于调试——它是 many-to-many 的 L 倍开销。

        pad_mask 在内部由 seq 构造后逐块下传（不是 forward 的参数）：每个块都要
        屏蔽，因为上一块的 FFN 会把 PAD 位置的残差改写为非零，仅屏蔽第一块不足以
        让 h 在 PAD 段保持为 0。
        """
        x = self.drop(self.item_emb(seq))
        pad_mask = (seq != self.pad).unsqueeze(-1).to(torch.float32)  # (B, L, 1)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.ln(x)
        if last_only:
            x = x[:, -1:]  # (B, 1, d)
        return x @ self.item_emb.weight[: self.n_items].T


class LRURec:
    """LRURec 包装类：数据/训练/早停/打分，接口与 ItemCF、BPR 完全一致。"""

    _EVAL_EVERY: int = 2  # 每 2 个 epoch 评一次 valid NDCG@10（单次约 3 秒）

    def __init__(
        self,
        embed_dim: int = 64,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 40,
        patience: int = 3,
        max_len: int = 50,
        n_blocks: int = 2,
        dropout: float = 0.2,
        seed: int = config.SEED,
    ) -> None:
        if embed_dim < 1 or batch_size < 1 or max_epochs < 1 or patience < 1 or max_len < 2:
            raise ValueError("embed_dim/batch_size/max_epochs/patience >= 1，max_len >= 2")
        if lr <= 0 or not 0.0 <= dropout < 1.0 or n_blocks < 1:
            raise ValueError("lr > 0，0 <= dropout < 1，n_blocks >= 1")
        self.embed_dim = embed_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.max_len = max_len
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.seed = seed

    # -----------------------------------------------------------
    # 训练
    # -----------------------------------------------------------
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "LRURec":
        """训练。valid 仅用于早停（fit(train) 的调用形式不变；不传则训满）。"""
        if train.duplicated(subset=["UserID", "MovieID"]).any():
            raise ValueError("训练集存在重复的 (UserID, MovieID)，请先聚合或去重")

        # ---- 索引：train 全集，不因正反馈过滤而缩减 ----
        self.users_ = np.sort(train["UserID"].unique())
        self.items_ = np.sort(train["MovieID"].unique())
        self.user_index_ = {u: i for i, u in enumerate(self.users_)}
        self.item_index_ = {m: i for i, m in enumerate(self.items_)}
        n_users, n_items = len(self.users_), len(self.items_)
        pad = n_items

        rows = train["UserID"].map(self.user_index_).to_numpy(dtype=np.int64)
        cols = train["MovieID"].map(self.item_index_).to_numpy(dtype=np.int64)
        ratings = train["Rating"].to_numpy()

        # top_k 屏蔽用：train 全部交互（任意评分）
        self._rated = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )

        # ---- 正例序列：只来自 train，按 (user, Timestamp, MovieID) 稳定排序 ----
        pos = ratings >= config.POSITIVE_RATING_THRESHOLD
        ts = train["Timestamp"].to_numpy()
        order = np.lexsort((cols[pos], ts[pos], rows[pos]))
        pu, pi = rows[pos][order], cols[pos][order]
        groups = np.split(pi, np.flatnonzero(np.diff(pu)) + 1)

        # 打分有效性：正例数 >= 1 才有历史可编码。零正例用户 _seq 全为 PAD，
        # 输出恒为 LN 偏置决定的常数向量，在评估里应一律未命中而非靠运气偶尔命中。
        # 注意判据不是 has_pair（>=2 才有训练对）：单正例用户没参与训练，
        # 但有 1 个 token，仍能给出有意义的打分。
        self._has_seq = np.zeros(n_users, dtype=bool)
        self._has_seq[np.unique(pu)] = True

        # ---- 两个序列视图 ----
        # _seq      : 最近 max_len 个正例（含最后一条），用于打分/预测"下一个"
        # train_in  : 去掉最后一条的历史，用于训练（target 是最后一条）
        self._seq = np.full((n_users, self.max_len), pad, dtype=np.int64)
        seq_in = np.full((n_users, self.max_len), pad, dtype=np.int64)
        seq_tg = np.full(n_users, pad, dtype=np.int64)
        has_pair = np.zeros(n_users, dtype=bool)
        for u, g in zip(np.unique(pu), groups):
            tail = g[-self.max_len :]
            self._seq[u, self.max_len - len(tail) :] = tail
            if len(g) >= 2:  # 至少 1 个历史 + 1 个目标
                w = g[-(self.max_len + 1) :]  # 最多 max_len 个历史 + 1 个目标
                seq_in[u, self.max_len - len(w) + 1 :] = w[:-1]
                seq_tg[u] = w[-1]
                has_pair[u] = True

        train_users = np.flatnonzero(has_pair)
        if len(train_users) == 0:
            raise ValueError("没有可训练的序列：所有用户的正反馈都不足 2 条")
        train_in = seq_in[train_users]
        train_tg = seq_tg[train_users]
        n_train = len(train_users)

        # 早停与外部评估共用：训练集索引只建一次，不必每轮 groupby 98 万行
        self.train_index_ = TrainIndex(train, self.items_)
        valid_splits = {"train": train, "test": valid} if valid is not None else None

        # ---- 模型：CPU，固定 seed 可复现 ----
        torch.manual_seed(self.seed)
        self._net = _LRUNet(n_items, self.embed_dim, self.embed_dim, self.n_blocks, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)

        best_ndcg = -np.inf
        best_state: dict | None = None
        bad_rounds = 0
        self.best_epoch_ = 0

        for epoch in range(1, self.max_epochs + 1):
            self._net.train()
            perm = rng.permutation(n_train)  # 每 epoch 打乱用户顺序
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                seq = torch.from_numpy(train_in[idx])
                tgt = torch.from_numpy(train_tg[idx])
                # many-to-one：只取最后位置的 logits，(B, n_items)
                logits = self._net(seq, last_only=True).squeeze(1)
                loss = F.cross_entropy(logits, tgt, ignore_index=pad)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                optimizer.step()

            # ---- 早停：只看 valid，每 _EVAL_EVERY 个 epoch 一次，最后一轮必评 ----
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
                    best_state = {
                        k: v.detach().clone() for k, v in self._net.state_dict().items()
                    }
                    bad_rounds = 0
                else:
                    bad_rounds += 1
                    if bad_rounds >= self.patience:
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        else:
            self.best_epoch_ = self.max_epochs
        return self

    # -----------------------------------------------------------
    # 打分与推荐
    # -----------------------------------------------------------
    @torch.no_grad()
    def _score_rows(self, rows: np.ndarray) -> np.ndarray:
        """按用户行号批量打分：取序列最后位置的 next-item logits。"""
        self._net.eval()
        out = np.empty((len(rows), len(self.items_)), dtype=np.float32)
        for s in range(0, len(rows), 512):
            seq = torch.from_numpy(self._seq[rows[s : s + 512]])
            out[s : s + 512] = self._net(seq, last_only=True).squeeze(1).numpy()
        # 零正例用户没有可编码的历史，一律置 -inf：评估时必然未命中，
        # top_k 的 np.isfinite 过滤会把它变成空推荐
        out[~self._has_seq[rows]] = -np.inf
        return out

    @torch.no_grad()
    def score_users(self, users=None) -> tuple[np.ndarray, np.ndarray]:
        """(users, scores)：dense float32，行对齐返回的 users；未知用户同步过滤。"""
        if users is None:
            users = self.users_
            rows = np.arange(len(self.users_), dtype=np.int64)
        else:
            users = np.asarray(users)
            keep = np.fromiter(
                (int(u) in self.user_index_ for u in users), dtype=bool, count=len(users)
            )
            users = users[keep]
            rows = np.array([self.user_index_[int(u)] for u in users], dtype=np.int64)
        return users, self._score_rows(rows)

    @torch.no_grad()
    def score_user(self, user_id: int) -> np.ndarray:
        """单用户打分向量（长度 = len(items_)）；未知用户返回全 0。"""
        user_id = int(user_id)
        if user_id not in self.user_index_:
            return np.zeros(len(self.items_), dtype=np.float32)
        return self._score_rows(np.array([self.user_index_[user_id]], dtype=np.int64))[0]

    def top_k(self, user_id: int, k: int = config.TOP_K) -> list[tuple[int, float]]:
        """[(MovieID, score)] 降序；屏蔽该用户在 train 中全部已交互电影。"""
        user_id = int(user_id)
        if user_id not in self.user_index_:
            return []

        # score_user 每次返回新分配的行（_score_rows 的 out 是临时数组，[0] 取的是
        # 它的视图），没有其他引用，原地屏蔽安全；若将来改成共享 buffer 需先 .copy()
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


# ---------------------------------------------------------------
# 网格搜索 + 评估（python -m src.models.lrurec）
# ---------------------------------------------------------------
def tune_and_evaluate() -> tuple[LRURec, pd.DataFrame, tuple[int, float]]:
    """4 组超参只在 valid 上选；test 仅对最佳组合评一次。"""
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

    grid = [(embed_dim, lr) for embed_dim in (32, 64) for lr in (1e-3, 5e-3)]

    rows: list[dict] = []
    best = None
    best_model: LRURec | None = None
    for n, (embed_dim, lr) in enumerate(grid, 1):
        model = LRURec(embed_dim=embed_dim, lr=lr).fit(
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
            f"  [{n}/{len(grid)}] embed_dim={embed_dim} lr={lr:g} "
            f"best_epoch={model.best_epoch_:2d}  NDCG@10={m['ndcg@10']:.4f}",
            flush=True,
        )
        rows.append(
            {
                "embed_dim": embed_dim,
                "lr": lr,
                "best_epoch": model.best_epoch_,
                "recall@10": round(m["recall@10"], 5),
                "ndcg@10": round(m["ndcg@10"], 5),
                "mrr@10": round(m["mrr@10"], 5),
            }
        )
        if best is None or m["ndcg@10"] > best[0]:
            best = (m["ndcg@10"], embed_dim, lr)
            best_model = model  # 固定 seed，复用等价于重训

    table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    print("\n== 验证集网格搜索（指标越高越好） ==")
    print(table.to_string(index=False))

    assert best_model is not None, "网格非空，必然选出最佳模型"
    _, best_dim, best_lr = best
    model = best_model
    print(f"\n最佳配置: embed_dim={best_dim}, lr={best_lr}")

    report_metrics(model, valid_splits, model.train_index_, "验证集（同一配置）")
    report_metrics(model, test_splits, model.train_index_, "测试集（最终报告口径）")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "lrurec_tuning.txt"
    out.write_text(
        table.to_string(index=False)
        + f"\n\n最佳: embed_dim={best_dim}, lr={best_lr}\n",
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out.relative_to(config.ROOT)}")

    # ---- 最终模型口径②：train+valid 重训后再评 test（与 ItemCF/BPR 并列）----
    # 没有独立 valid 可早停，固定按最佳模型的 best_epoch 轮数训练（同 BPR 的局限）
    full_train = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    final = LRURec(
        embed_dim=best_dim, lr=best_lr, max_epochs=model.best_epoch_
    ).fit(full_train)
    report_metrics(
        final,
        {"train": full_train, "test": splits["test"]},
        final.train_index_,
        "测试集（train+valid 重训）",
    )

    # 示例推荐用口径②的 final（最终交付模型，见过 train+valid，屏蔽集同样是 full_train）
    movie_title = movies.set_index("MovieID")[["Title", "Genres"]]
    print("\n== 示例推荐：UserID=1（Top 5，train+valid 重训模型，已交互已屏蔽）==")
    for movie_id, score in final.top_k(1, 5):
        title, genres = movie_title.loc[movie_id]
        print(f"  {score:7.3f}  {movie_id:4d}  {title}  [{genres}]")

    return model, table, (best_dim, best_lr)


if __name__ == "__main__":
    tune_and_evaluate()
