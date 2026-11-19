"""训练三个算法的最终交付模型并写入磁盘缓存，供交互界面秒开加载。

同一份数据 + 同一组超参重复运行会直接命中缓存（打印 hit），不会重训。
缓存布局与指纹规则见 src/cache.py。

最终交付模型使用 **train + valid 全量重训**（与评测口径②一致）：
界面演示的是"见过最多数据"的那个模型。

用法
----
    python -m src.persist              # 处理全部三个模型
    python -m src.persist lrurec       # 只处理指定模型

产物
----
    models/<模型>_<超参哈希>_<数据指纹>/{meta.json, state.npz, *.sparse.npz}
    已被 .gitignore 的 `/models/` 与 `*.pkl` 规则排除，不随仓库分发。
    换 torch / pandas 版本后建议删除 models/ 重跑。
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from . import cache, config
from .data_loader import load_all
from .models.bpr import BPR
from .models.itemcf import ItemCF
from .models.lrurec import LRURec
from .split import leave_one_out_split, load_splits

# 验证集网格搜索选出的最优超参（见 outputs/results/*_tuning.txt）
#   ItemCF  weighting=binary, alpha=0, k_neighbors=20          （48 组网格）
#   BPR     embed_dim=32, lr=5e-3, reg=1e-5，早停于 epoch 32  （8 组网格）
#   LRURec  embed_dim=64, lr=1e-3，早停于 epoch 10           （4 组网格）
# 第三项为重训轮数：重训时没有独立 valid 可早停，固定用验证集选出的轮数；ItemCF 无轮数
SPECS: dict[str, tuple[type, dict, int | None]] = {
    "itemcf": (ItemCF, {"k_neighbors": 20, "alpha": 0.0, "weighting": "binary"}, None),
    "bpr": (BPR, {"embed_dim": 32, "lr": 5e-3, "reg": 1e-5}, 32),
    "lrurec": (LRURec, {"embed_dim": 64, "lr": 1e-3}, 10),
}


def main(argv: list[str]) -> int:
    unknown = [name for name in argv if name not in SPECS]
    if unknown:
        print(f"未知模型：{', '.join(unknown)}；可选：{', '.join(SPECS)}")
        return 2
    names = argv or list(SPECS)

    # 只加载一次：兜底划分与重训共用
    ratings, _, _ = load_all()
    try:
        splits = load_splits()
    except FileNotFoundError:
        splits = leave_one_out_split(ratings)

    full_train = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    print(
        f"重训数据：train {len(splits['train'])} + valid {len(splits['valid'])} "
        f"= {len(full_train)} 行（数据指纹 {cache.data_fingerprint(full_train)}）",
        flush=True,
    )

    for name in names:
        cls, params, epochs = SPECS[name]
        kwargs = dict(params)
        if epochs is not None:
            kwargs["max_epochs"] = epochs

        probe = cls(**kwargs)  # 不训练，只用来取完整生效超参（含默认 seed）
        hparams = probe.hparams()
        hit = cache.load(cls, full_train, **hparams)
        if hit is not None:
            meta = hit.cache_meta_
            print(
                f"  {name:7s} hit   {meta['trained_at']}  "
                f"users={meta['n_users']} items={meta['n_items']}  "
                f"{cache.cache_dir(name, hparams, full_train).name}",
                flush=True,
            )
            continue

        t0 = time.time()
        model = probe.fit(full_train)  # 不传 valid：重训阶段不做早停
        elapsed = time.time() - t0
        path = cache.save(model, full_train, extra={"fit_seconds": round(elapsed, 1)})
        size_mb = sum(f.stat().st_size for f in path.iterdir()) / 2**20
        print(
            f"  {name:7s} miss  {elapsed:6.1f}s  "
            f"users={len(model.users_)} items={len(model.items_)}  "
            f"{size_mb:6.1f} MB  -> {path.relative_to(config.ROOT)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
