"""Streamlit 交互界面：选用户 -> 看推荐 -> 看历史。

启动（项目根目录）：
    streamlit run src/app.py

模型来自 models/ 的磁盘缓存（由 python -m src.persist 生成），
界面只做加载与推理：切换用户、切换算法都不会触发训练。
"""

from __future__ import annotations

import html as html_lib
import sys
from pathlib import Path

# `streamlit run src/app.py` 只把 src/ 放进 sys.path，项目根要自己补，
# 否则 `from src import service` 会 ImportError。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src import config, service

st.set_page_config(
    page_title="MovieLens 电影推荐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# 样式：只改观感，不动组件行为；类名统一 mlrec- 前缀，避免与 Streamlit 内部样式冲突
# ---------------------------------------------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

.mlrec-hero { padding: 2px 0 4px; margin-bottom: 18px; }
.mlrec-hero h1 {
  color: #17171b; font-size: 26px; margin: 0 0 6px; letter-spacing: .2px; font-weight: 700;
}
.mlrec-hero p { color: #6b6b76; font-size: 13.5px; margin: 0; line-height: 1.65; }
.mlrec-chip {
  display: inline-block; margin: 12px 8px 0 0; padding: 2px 10px;
  border-radius: 999px; background: #f5f5f8; border: 1px solid #e6e6ec; color: #4f4f5a;
  font-size: 12px; letter-spacing: .2px;
}

.mlrec-card {
  display: flex; align-items: center; gap: 14px;
  background: #fff; border: 1px solid #ececf1; border-radius: 12px;
  padding: 10px 14px; margin-bottom: 8px;
  box-shadow: 0 1px 2px rgba(16, 16, 20, .04);
}
.mlrec-card:hover { border-color: #dcdce4; box-shadow: 0 4px 14px rgba(16, 16, 20, .08); }
.mlrec-badge {
  flex: none; width: 26px; height: 26px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: #6b6b76; background: #f2f2f6;
}
.mlrec-badge.is-top { background: #e50914; color: #fff; }
.mlrec-main { flex: 1; min-width: 0; }
.mlrec-title {
  font-size: 15px; font-weight: 600; color: #17171b;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mlrec-sub {
  font-size: 12px; color: #8b8b95; margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mlrec-score {
  flex: none; font-size: 15px; font-weight: 700; color: #e50914;
  font-variant-numeric: tabular-nums;
}
.mlrec-bar {
  flex: none; width: 72px; height: 6px; border-radius: 3px;
  background: #f0f0f4; overflow: hidden;
}
.mlrec-bar > i { display: block; height: 100%; background: #e50914; border-radius: 3px; }
.mlrec-date {
  flex: none; font-size: 12px; color: #9a9aa4;
  white-space: nowrap;   /* 不能给固定宽度：'4/5  2001-01-06' 会被挤成两行 */
}

/* 指标卡：Streamlit 原生 st.metric 无背景，这里补成白底圆角卡片。
   min-height 取 110px：用户区最高的"人群"卡（36px 字号折行）是 108px，
   取 110 才能让同排四张底边对齐 */
[data-testid="stMetric"] {
  background: #fff; border: 1px solid #ececf1; border-radius: 12px;
  padding: 12px 16px; min-height: 110px;
  box-shadow: 0 1px 2px rgba(16, 16, 20, .04);
}
/* 真正画边框的是 expander 内部的 <details>，外层 div 只是容器 */
[data-testid="stExpander"] details {
  border: 1px solid #ececf1;
  border-radius: 12px;
  background: #fff;
}
/* summary 自身圆角是 8px，比外框 12px 小，收起态会露出方角 */
[data-testid="stExpander"] summary { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 渲染辅助
# ---------------------------------------------------------------
def render_rank_cards(df: pd.DataFrame, show_genres: bool = True) -> None:
    """把推荐表渲染成卡片：前三名用主色徽章，得分右对齐等宽数字。"""
    if df.empty:
        st.caption("（无推荐：该用户在训练数据中缺少正反馈）")
        return

    cards = []
    for _, row in df.iterrows():
        badge = "mlrec-badge is-top" if row["排名"] <= 3 else "mlrec-badge"
        sub = (
            f'<div class="mlrec-sub">{html_lib.escape(str(row["题材"]))}</div>'
            if show_genres
            else ""
        )
        cards.append(
            f'<div class="mlrec-card">'
            f'<div class="{badge}">{int(row["排名"])}</div>'
            f'<div class="mlrec-main">'
            f'<div class="mlrec-title">{html_lib.escape(str(row["片名"]))}</div>{sub}'
            f"</div>"
            f'<div class="mlrec-score">{row["得分"]:.2f}</div>'
            f"</div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_history_cards(df: pd.DataFrame) -> None:
    """历史记录用评分条代替数字，一眼看出喜好强度。"""
    if df.empty:
        st.caption("该用户在 train+valid 中没有评分记录。")
        return

    cards = []
    for _, row in df.iterrows():
        score = int(row["评分"])
        cards.append(
            f'<div class="mlrec-card">'
            f'<div class="mlrec-main">'
            f'<div class="mlrec-title">{html_lib.escape(str(row["片名"]))}</div>'
            f'<div class="mlrec-sub">{html_lib.escape(str(row["题材"]))}</div>'
            f"</div>"
            f'<div class="mlrec-bar" title="评分 {score}/5">'
            f'<i style="width:{score * 20}%"></i></div>'
            f'<div class="mlrec-date">{score}/5　{row["日期"]}</div>'
            f"</div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


# ---------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------
side = st.sidebar
side.markdown("### 配置")

name = side.selectbox(
    "推荐算法",
    service.MODEL_NAMES,
    format_func=lambda n: service.LABELS[n],
)

users = service.user_ids()
if "uid" not in st.session_state:
    st.session_state["uid"] = int(users[0])
if side.button("随机换一个用户", width="stretch"):
    st.session_state["uid"] = int(users[np.random.randint(len(users))])
    st.rerun()

uid = side.number_input(
    "UserID", min_value=int(users.min()), max_value=int(users.max()), step=1, key="uid"
)
k = side.slider("推荐数量 Top-K", min_value=5, max_value=20, value=config.TOP_K)
compare = side.toggle("三模型并排对比", value=False)

# ---------------------------------------------------------------
# 数据准备（模型缺失时给可操作报错，不在请求路径上训练）
# ---------------------------------------------------------------
try:
    rec = service.recommend(name, uid, k)
    meta = service.model_meta(name)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if not service.known_user(uid):
    st.markdown(
        '<div class="mlrec-hero"><h1>MovieLens 电影推荐</h1>'
        f"<p>UserID {uid} 不在训练数据中，属于冷启动用户，"
        "三个模型都无法给出个性化推荐。</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

profile = service.user_profile(uid)

st.markdown(
    f"""
<div class="mlrec-hero">
  <h1>MovieLens 电影推荐</h1>
  <p>ItemCF / BPR / LRURec 三个算法在 train+valid 上训练并落盘，界面只做加载与推理。</p>
  <span class="mlrec-chip">{html_lib.escape(service.LABELS[name])}</span>
  <span class="mlrec-chip">UserID {uid}</span>
  <span class="mlrec-chip">Top {k}</span>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("已评分电影", f"{profile['n_history']}", help="train+valid 中的评分条数，test 不参与")
c2.metric("平均评分", f"{profile['mean_rating']:.2f}")
c3.metric("评分 >= 4", f"{profile['n_positive']}", help="视作正反馈，BPR / LRURec 只用这部分训练")
c4.metric("人群", f"{profile['gender']} · {profile['age']}")
st.caption(
    f"职业：{profile['occupation']}　|　活跃区间：{profile['first_date']} ~ {profile['last_date']}"
)

st.markdown(f"#### Top {k} 推荐")
st.caption(f"已屏蔽该用户看过的 {profile['n_history']} 部电影；得分只在同一模型内可比")
render_rank_cards(rec)

with st.expander(f"最近评分的电影（共 {profile['n_history']} 条，显示最近 12 条）"):
    render_history_cards(service.history(uid, limit=12))

# ---------------------------------------------------------------
# 离线指标：delta 相对流行度基线，说明"学到东西没有"
# ---------------------------------------------------------------
m = service.METRICS[name]
base = service.BASELINE

st.markdown("#### 离线表现")
st.caption("test 集（每用户时序最后 1 条），模型在 train+valid 上重训；delta 为相对流行度基线的增益")
d1, d2, d3 = st.columns(3)
d1.metric("Recall@10", f"{m['recall@10']:.4f}", f"{m['recall@10'] - base['recall@10']:+.4f}")
d2.metric("NDCG@10", f"{m['ndcg@10']:.4f}", f"{m['ndcg@10'] - base['ndcg@10']:+.4f}")
d3.metric("MRR@10", f"{m['mrr@10']:.4f}", f"{m['mrr@10'] - base['mrr@10']:+.4f}")
st.caption(
    f"最佳 epoch：{meta['best_epoch'] or '不适用（ItemCF 无梯度训练）'}　|　"
    f"缓存训练于 {meta['trained_at']}，fit {meta.get('fit_seconds', 0)} s"
)

# ---------------------------------------------------------------
# 三模型并排对比
# ---------------------------------------------------------------
if compare:
    st.markdown("#### 同一用户，三个模型的 Top 5")
    cols = st.columns(3, gap="medium")
    for col, other in zip(cols, service.MODEL_NAMES):
        with col:
            st.markdown(f"**{service.LABELS[other].split(' · ')[0]}**")
            render_rank_cards(service.recommend(other, uid, 5), show_genres=False)
    st.caption(
        "三个模型目标不同：ItemCF 按物品共现相似度聚合用户历史，BPR 学用户与物品隐向量的内积，"
        "LRURec 只依据最近一次行为预测下一部。目标不同，Top-K 重合度本就偏低；"
        "得分尺度也不同，跨模型比较分数没有意义，只能看各自内部的排序。"
    )

# ---------------------------------------------------------------
# 侧边栏底部：缓存来源，便于确认页面用的是哪次训练产物
# ---------------------------------------------------------------
side.divider()
side.caption(f"{meta['n_users']} 用户 / {meta['n_items']} 部电影")
side.caption("产物由 `python -m src.persist` 生成，切换算法或用户不会重训。")
