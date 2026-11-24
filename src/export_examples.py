"""导出报告 5.2 节的三模型 Top-K 示例（指定用户）。

与 ``analysis_overlap`` 一样走 ``src.service.recommend``，屏蔽口径与界面一致，
因此报告表格里的每一条都可以由本脚本原样复现。

运行：python -m src.export_examples
输出：outputs/results/examples.txt
"""

from __future__ import annotations

from pathlib import Path

from . import config, service

USERS = (1, 6040)
K = 5


def render(user_id: int, k: int = K) -> list[str]:
    lines = [f"== UserID {user_id} =="]
    for name in service.MODEL_NAMES:
        df = service.recommend(name, user_id, k)
        lines.append(f"[{name}]")
        for _, row in df.iterrows():
            lines.append(f"  {int(row['排名'])}. {row['片名']}（{row['题材']}）")
    return lines


def main() -> str:
    blocks = [render(user_id) for user_id in USERS]
    text = "\n\n".join("\n".join(block) for block in blocks)
    print(text)
    return text


if __name__ == "__main__":
    out_path: Path = config.RESULTS_DIR / "examples.txt"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(main() + "\n", encoding="utf-8")
    print(f"\n[已写入 {out_path.relative_to(config.ROOT)}]")
