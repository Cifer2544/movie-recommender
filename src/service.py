"""界面层的服务接口：只做加载、取数与格式化，不含任何训练逻辑。

三条职责边界
------------
- 模型一律从 models/ 的磁盘缓存加载（见 src/cache.py）。缓存未命中直接报错并
  提示先跑 ``python -m src.persist``；绝不在请求路径上现场训练，否则界面每次
  冷启动都要等几分钟。
- 历史与屏蔽口径统一为 **train + valid**（模型实际训练的数据），不读 test，
  因此界面里"已评分电影"与推荐屏蔽与离线评估口径一致。
- 返回给界面的 DataFrame 已带好片名与题材，app 层不再碰原始表。

缓存策略
--------
函数级 ``lru_cache``：Streamlit 每次交互会重跑脚本，但进程不变，因此模型对象
与数据表只在进程内加载一次。改代码触发热重载后进程会重建，需要重新加载
（三个模型合计约 1-2 秒，可接受）。
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import cache, config
from .data_loader import load_movies, load_users
from .models.bpr import BPR
from .models.itemcf import ItemCF
from .models.lrurec import LRURec
from .split import load_splits

# 交付超参必须与 src/persist.py 的 SPECS 一致，否则缓存指纹对不上，永远 miss。
DELIVERED: dict[str, tuple[type, dict]] = {
    "itemcf": (ItemCF, {"k_neighbors": 20, "alpha": 0.0, "weighting": "binary"}),
    "bpr": (BPR, {"embed_dim": 32, "lr": 5e-3, "reg": 1e-5, "max_epochs": 32}),
    "lrurec": (LRURec, {"embed_dim": 64, "lr": 1e-3, "max_epochs": 10}),
}

MODEL_NAMES: list[str] = list(DELIVERED)

LABELS = {
    "itemcf": "ItemCF · 物品协同过滤（课程算法）",
    "bpr": "BPR · 贝叶斯个性化排序（论文复现）",
    "lrurec": "LRURec · 线性循环单元序列模型（WSDM 2024）",
}

# 离线评估：test 集，模型在 train+valid 上重训（README 2.1 同源，由 python -m src.models.* 实测）
METRICS = {
    "itemcf": {"recall@10": 0.0452, "ndcg@10": 0.0215, "mrr@10": 0.0143},
    "bpr": {"recall@10": 0.0679, "ndcg@10": 0.0330, "mrr@10": 0.0226},
    "lrurec": {"recall@10": 0.0831, "ndcg@10": 0.0471, "mrr@10": 0.0363},
}

# 流行度基线（train），README 2.1 同源：界面用它给各算法算"相对基线增益"
BASELINE = {"recall@10": 0.0351, "ndcg@10": 0.0174, "mrr@10": 0.0122}

# 年龄段用短标签：指标卡在窄窗口下只有约 150px，长文本会被省略号截断
AGE_LABELS = {
    1: "<18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

# ml-1m/README 中的 Occupation 编码
OCCUPATION_LABELS = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

# movies.dat 里有 807 条片名采用"冠词后置"的目录排序写法（如
# ``Shawshank Redemption, The (1994)``，让 The Matrix 排在 M 而不是 T）。
# 这是数据源的原样内容、不是读取错误，但直接展示容易被误认为 bug，
# 因此只在**显示层**把冠词还原到前面；MovieID 匹配与指标计算都用原始列。
_INVERTED_TITLE = re.compile(r"^(.+), (The|A|An)( \(\d{4}\))?$")


def display_title(title: str) -> str:
    """把 ``Xxx, The (1994)`` 还原为 ``The Xxx (1994)``；不符合模式的原样返回。"""
    match = _INVERTED_TITLE.match(str(title))
    if match is None:
        return str(title)
    main, article, year = match.groups()
    return f"{article} {main}{year or ''}"


# ---------------------------------------------------------------
# 静态数据
# ---------------------------------------------------------------
@lru_cache(maxsize=1)
def full_train() -> pd.DataFrame:
    """模型实际训练的数据（train + valid）；界面展示历史与屏蔽都以此为准。"""
    try:
        splits = load_splits()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "未找到 outputs/splits/*.parquet，请先在项目根目录运行：python -m src.split"
        ) from exc
    return pd.concat([splits["train"], splits["valid"]], ignore_index=True)


@lru_cache(maxsize=1)
def movies() -> pd.DataFrame:
    """片名表，按 MovieID 建索引，只留界面要显示的三列。"""
    return load_movies().set_index("MovieID")[["Title", "Genres", "Year"]]


@lru_cache(maxsize=1)
def user_ids() -> np.ndarray:
    return np.sort(full_train()["UserID"].unique())


@lru_cache(maxsize=1)
def _user_id_set() -> frozenset[int]:
    return frozenset(int(u) for u in user_ids())


@lru_cache(maxsize=1)
def _history_by_user() -> dict[int, pd.DataFrame]:
    """按用户切分一次历史，之后查单个用户只做字典取值。"""
    ordered = full_train().sort_values(["UserID", "Timestamp"])
    return {int(u): g for u, g in ordered.groupby("UserID", sort=False)}


@lru_cache(maxsize=1)
def _users_table() -> pd.DataFrame:
    return load_users().set_index("UserID")


# ---------------------------------------------------------------
# 模型
# ---------------------------------------------------------------
@lru_cache(maxsize=None)
def get_model(name: str):
    """从磁盘缓存加载交付模型；未命中直接报错，不在请求路径上训练。"""
    if name not in DELIVERED:
        raise KeyError(f"未知模型 {name}，可选：{', '.join(DELIVERED)}")
    cls, params = DELIVERED[name]
    model = cache.load(cls, full_train(), **cls(**params).hparams())
    if model is None:
        raise FileNotFoundError(
            f"未找到 {name} 的磁盘缓存，请先在项目根目录运行：python -m src.persist {name}"
        )
    return model


def model_meta(name: str) -> dict:
    """缓存目录 meta.json：训练时间、best_epoch、数据规模、落盘耗时。"""
    return get_model(name).cache_meta_


def known_user(user_id: int) -> bool:
    """是否出现在训练数据中；不出现则属于冷启动用户，模型无法个性化。"""
    return int(user_id) in _user_id_set()


# ---------------------------------------------------------------
# 展示用数据
# ---------------------------------------------------------------
def user_profile(user_id: int) -> dict:
    """用户属性 + 历史统计（来自 train + valid）。"""
    user_id = int(user_id)
    hist = _history_by_user().get(user_id)

    info = _users_table().loc[user_id] if user_id in _users_table().index else None
    return {
        "gender": ("男" if info["Gender"] == "M" else "女") if info is not None else "未知",
        "age": AGE_LABELS.get(int(info["Age"]), "未知") if info is not None else "未知",
        "occupation": (
            OCCUPATION_LABELS.get(int(info["Occupation"]), "unknown")
            if info is not None
            else "未知"
        ),
        "n_history": 0 if hist is None else len(hist),
        "mean_rating": 0.0 if hist is None else float(hist["Rating"].mean()),
        "n_positive": (
            0
            if hist is None
            else int((hist["Rating"] >= config.POSITIVE_RATING_THRESHOLD).sum())
        ),
        "first_date": (
            "" if hist is None else pd.to_datetime(hist["Timestamp"].min(), unit="s").strftime("%Y-%m-%d")
        ),
        "last_date": (
            "" if hist is None else pd.to_datetime(hist["Timestamp"].max(), unit="s").strftime("%Y-%m-%d")
        ),
    }


def history(user_id: int, limit: int = 12) -> pd.DataFrame:
    """该用户在 train+valid 中的交互，按时间倒序取最近 limit 条。

    不单列年份：Title 本身形如 ``Toy Story (1995)``，再放一列会让窄屏横向滚动。
    """
    hist = _history_by_user().get(int(user_id))
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["片名", "题材", "评分", "日期"])

    out = hist.sort_values("Timestamp", ascending=False).head(limit).join(movies(), on="MovieID")
    out["片名"] = out["Title"].map(display_title)
    out["题材"] = out["Genres"].str.replace("|", " / ", regex=False)
    out["日期"] = pd.to_datetime(out["Timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    return out.rename(columns={"Rating": "评分"})[
        ["片名", "题材", "评分", "日期"]
    ].reset_index(drop=True)


def recommend(name: str, user_id: int, k: int = config.TOP_K) -> pd.DataFrame:
    """Top-K 推荐表（排名 / 片名 / 题材 / 得分），已屏蔽该用户的历史。"""
    items = get_model(name).top_k(int(user_id), int(k))
    columns = ["排名", "片名", "题材", "得分"]
    if not items:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(items, columns=["MovieID", "得分"])
    df.insert(0, "排名", np.arange(1, len(df) + 1))
    df = df.join(movies(), on="MovieID")
    df["片名"] = df["Title"].map(display_title)
    df["题材"] = df["Genres"].str.replace("|", " / ", regex=False)
    return df[columns]
