"""模型缓存：按「超参 + 训练数据」指纹命名落盘，避免重复训练。

为什么不用 pickle 整个模型对象
------------------------------
pickle 会把类的模块路径写进文件（如 src.models.bpr.BPR）。一旦重构类名或移动文件，
旧缓存全部加载失败，报错信息是一串 "Can't get attribute ..."，很难定位。
这里只存 state（权重 / 相似度矩阵 / 序列索引），加载时先 ``__init__(**hparams)``
再 ``set_state()``，代码重构不影响已有缓存。

缓存布局
--------
    models/<模型>_<超参哈希>_<数据指纹>/
        meta.json          超参、数据规模、best_epoch、训练时间（指标可回填）
        state.npz          所有 ndarray 形式的 state（torch 权重也转成 ndarray）
        <key>.sparse.npz   每个 scipy 稀疏矩阵单独存（csr 压缩后体积小得多）

为什么用内容指纹做目录名
------------------------
命名成 ``*_latest`` 必然出现"改了超参或数据却读到旧缓存"的静默错误。
超参指纹把构造参数排序后取 blake2b，seed 也在其中，所以"改了 max_len 就该重训"是自动的。
数据指纹取行数 + 四列的和与唯一值数：不用文件 mtime（重新生成同名 parquet
内容相同也会变），也不用完整哈希（99 万行要几百毫秒）。

每个模型需要实现三个方法
------------------------
    hparams()               -> dict   影响训练结果的全部构造参数（必须含 seed）
    get_state()             -> dict   {名字: ndarray 或 csr_matrix}，只含推理所需
    set_state(state, meta)            按 state 恢复，自己重建索引 dict

推理不需要的内容一律不存：BPR 的 ``_pos_mask``（负采样用）、各模型的
``train_index_``（evaluate_scores 收到 None 时会自行构建）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import config

_FINGERPRINT_COLUMNS = ("UserID", "MovieID", "Rating", "Timestamp")
_SPARSE_SUFFIX = ".sparse.npz"


def data_fingerprint(train: pd.DataFrame) -> str:
    """训练数据摘要：行数 + 四列的和与唯一值数，足以区分 train / train+valid。"""
    parts = [f"n={len(train)}"]
    for col in _FINGERPRINT_COLUMNS:
        values = train[col].to_numpy()
        parts.append(f"{col}={int(values.sum())}:{len(np.unique(values))}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=8).hexdigest()


def params_fingerprint(hparams: dict) -> str:
    """超参摘要：排序后序列化再取短 hash，seed 参与其中。"""
    text = json.dumps(hparams, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=6).hexdigest()


def cache_dir(model_name: str, hparams: dict, train: pd.DataFrame) -> Path:
    tag = f"{params_fingerprint(hparams)}_{data_fingerprint(train)}"
    return config.MODELS_DIR / f"{model_name}_{tag}"


def _save_state(state: dict, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if sparse.issparse(value):
            sparse.save_npz(path / f"{key}{_SPARSE_SUFFIX}", value.tocsr(), compressed=True)
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path / "state.npz", **arrays)


def _load_state(path: Path) -> dict:
    state: dict = {}
    with np.load(path / "state.npz") as archive:
        for key in archive.files:
            state[key] = archive[key]
    for file in path.glob(f"*{_SPARSE_SUFFIX}"):
        state[file.name[: -len(_SPARSE_SUFFIX)]] = sparse.load_npz(file)
    return state


def _validate(model, train: pd.DataFrame, path: Path) -> None:
    """物品表必须与当前训练数据一致；不一致说明缓存与数据已脱钩，直接报错。"""
    expected = np.sort(train["MovieID"].unique())
    if not np.array_equal(model.items_, expected):
        raise ValueError(
            f"缓存物品表与训练数据不一致（缓存 {len(model.items_)} 部，"
            f"当前 {len(expected)} 部）：请删除 {path} 后重跑"
        )


def save(model, train: pd.DataFrame, extra: dict | None = None) -> Path:
    """把 model 的 state 与 meta 写入缓存目录，返回目录路径。"""
    hparams = model.hparams()
    path = cache_dir(model.__class__.__name__.lower(), hparams, train)
    path.mkdir(parents=True, exist_ok=True)
    _save_state(model.get_state(), path)

    meta = {
        "model": model.__class__.__name__,
        "hparams": hparams,
        "data_fingerprint": data_fingerprint(train),
        "n_train_rows": int(len(train)),
        "n_users": int(len(model.users_)),
        "n_items": int(len(model.items_)),
        "best_epoch": int(getattr(model, "best_epoch_", 0) or 0),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta.update(extra or {})
    (path / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def load(cls, train: pd.DataFrame, **hparams):
    """命中返回已恢复的模型（并挂上 ``model.cache_meta_``），未命中返回 None。"""
    path = cache_dir(cls.__name__.lower(), hparams, train)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = cls(**meta["hparams"])
    model.set_state(_load_state(path), meta)
    _validate(model, train, path)
    model.cache_meta_ = meta
    return model
