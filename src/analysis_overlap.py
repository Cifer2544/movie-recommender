"""推荐内容分析：模型间重合度、覆盖度与新颖度（对应报告 5.3 / 5.4）。

三项统计统一走 ``src.service.recommend``，因此屏蔽口径（该用户全部评分历史）
与界面、离线评估完全一致；抽样用户数固定为 300、随机种子取 ``config.SEED``，
结果写入 outputs/results/overlap.txt，报告中的数字由此文件复现。

运行：python -m src.analysis_overlap
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from . import config, service

N_USERS = 300
K = 10


def _title_to_id() -> dict[str, int]:
    """service 返回的表只有片名，这里还原为 MovieID 以便集合运算。"""
    table = service.movies()
    return {
        service.display_title(title): int(movie_id)
        for movie_id, title in table["Title"].items()
    }


def collect(users: np.ndarray, k: int = K) -> dict[str, dict[int, list[int]]]:
    """对抽样用户逐个模型取 Top-K 的 MovieID 列表。"""
    mapper = _title_to_id()
    recs: dict[str, dict[int, list[int]]] = {name: {} for name in service.MODEL_NAMES}
    for user_id in users:
        user_id = int(user_id)
        for name in service.MODEL_NAMES:
            df = service.recommend(name, user_id, k)
            recs[name][user_id] = [mapper[title] for title in df["片名"]]
    return recs


def overlap_table(recs: dict[str, dict[int, list[int]]], k: int = K) -> list[dict]:
    """两两模型在每个用户上的 Top-K 重合条数。"""
    rows = []
    for a, b in itertools.combinations(service.MODEL_NAMES, 2):
        counts = [
            len(set(recs[a][uid]) & set(recs[b][uid]))
            for uid in recs[a]
        ]
        rows.append({
            "pair": f"{a}-{b}",
            "mean_overlap": float(np.mean(counts)),
            "no_overlap_share": float(np.mean([c == 0 for c in counts])),
            "k": k,
        })
    return rows


def coverage_table(recs: dict[str, dict[int, list[int]]], k: int = K) -> list[dict]:
    """覆盖度、重复推荐比例、被推荐电影的平均评分人数与新颖度。"""
    popularity = service.popularity_scores()
    n_train = len(service.full_train())
    n_users = len(next(iter(recs.values())))
    rows = []
    for name in service.MODEL_NAMES:
        flat = [mid for items in recs[name].values() for mid in items]
        unique = len(set(flat))
        counts = popularity.reindex(flat).to_numpy(dtype=float)
        novelty = float(np.mean(-np.log2(counts / n_train)))
        rows.append({
            "model": name,
            "unique_movies": unique,
            "repeat_share": 1.0 - unique / (n_users * k),
            "coverage": unique / len(popularity),
            "mean_popularity": float(np.mean(counts)),
            "mean_novelty": novelty,
            "n_users": n_users,
            "k": k,
        })
    return rows


def format_report(overlap: list[dict], coverage: list[dict]) -> str:
    lines = [
        f"n_users={coverage[0]['n_users']} seed={config.SEED} k={coverage[0]['k']}",
        "",
        "[两两 Top-K 重合度]",
        f"{'model_pair':<16}{'mean_overlap':>14}{'no_overlap_share':>18}",
    ]
    for row in overlap:
        lines.append(
            f"{row['pair']:<16}{row['mean_overlap']:>14.4f}{row['no_overlap_share']:>18.4f}"
        )
    lines += [
        "",
        "[覆盖度、流行度与新颖度]",
        f"{'model':<10}{'unique':>8}{'repeat':>9}{'coverage':>10}"
        f"{'mean_pop':>10}{'novelty':>9}",
    ]
    for row in coverage:
        lines.append(
            f"{row['model']:<10}{row['unique_movies']:>8d}{row['repeat_share']:>9.4f}"
            f"{row['coverage']:>10.4f}{row['mean_popularity']:>10.1f}"
            f"{row['mean_novelty']:>9.4f}"
        )
    return "\n".join(lines)


def main() -> str:
    rng = np.random.default_rng(config.SEED)
    users = np.sort(rng.choice(service.user_ids(), size=N_USERS, replace=False))
    recs = collect(users)
    text = format_report(overlap_table(recs), coverage_table(recs))
    print(text)
    return text


if __name__ == "__main__":
    out_path: Path = config.RESULTS_DIR / "overlap.txt"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(main() + "\n", encoding="utf-8")
    print(f"\n[已写入 {out_path.relative_to(config.ROOT)}]")
