"""数据集信息提取（对应评分项 "Dataset information"）。

输出 7 项内容：
    1. 各表 shape
    2. 空值总数
    3. 存在空值的列
    4. comparison dataset（三表合并的建模宽表 + 题材 multi-hot）
    5. DataFrame.info()
    6. 每列的唯一值个数
    7. 每列的数据类型

运行：python -m src.dataset_info
控制台输出同时写入 outputs/results/dataset_info.txt（该目录已被 git 忽略）。
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from . import config
from .data_loader import load_all

LINE = "=" * 72


def build_comparison_dataset(ratings, movies, users):
    """构建 comparison dataset：一份可供三个模型共用的建模宽表。

    三表按 UserID / MovieID 左连接，得到"每条评分 + 用户画像 + 电影信息"的
    长表；题材拆成 18 个 0/1 列（multi-hot），方便后续内容类特征与分组分析。
    该表只描述已观测的 (用户, 电影) 对，不含任何由标签推导出的统计量，
    因此不会把验证/测试标签泄漏进训练特征。
    """
    genres = sorted({g for items in movies["GenreList"] for g in items})

    df = (
        ratings.merge(
            movies[["MovieID", "Title", "TitleClean", "Year", "Genres", "GenreList"]],
            on="MovieID",
            how="left",
        )
        .merge(users, on="UserID", how="left")
    )

    for g in genres:
        df[f"genre_{g}"] = df["GenreList"].apply(lambda items, g=g: int(g in items))
    df = df.astype({f"genre_{g}": "int8" for g in genres})

    return df, genres


def section(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


def main():
    ratings, movies, users = load_all()

    section("1. shape")
    print(f"ratings          : {ratings.shape}")
    print(f"movies           : {movies.shape}")
    print(f"users            : {users.shape}")

    section("2. 空值总数（三张原始表）")
    for name, df in (("ratings", ratings), ("movies", movies), ("users", users)):
        print(f"{name:9s}: {int(df.isna().sum().sum())}")

    section("3. 存在空值的列")
    for name, df in (("ratings", ratings), ("movies", movies), ("users", users)):
        null_cols = df.columns[df.isna().any()].tolist()
        detail = (
            "无空值列"
            if not null_cols
            else ", ".join(f"{c}({int(df[c].isna().sum())})" for c in null_cols)
        )
        print(f"{name:9s}: {detail}")

    section("4. comparison dataset")
    comparison, genres = build_comparison_dataset(ratings, movies, users)
    print(f"shape            : {comparison.shape}")
    print(f"题材列（{len(genres)} 个）: {', '.join(genres)}")
    print(f"空值总数         : {int(comparison.isna().sum().sum())}")
    print("\n前 3 行（关键列）：")
    cols = ["UserID", "MovieID", "TitleClean", "Year", "Rating", "Gender", "Age", "genre_Drama"]
    print(comparison[cols].head(3).to_string(index=False))

    section("5. DataFrame.info()：ratings")
    ratings.info()

    section("5. DataFrame.info()：movies")
    movies.drop(columns=["GenreList"]).info()

    section("5. DataFrame.info()：users")
    users.info()

    section("5. DataFrame.info()：comparison dataset")
    comparison.drop(columns=["GenreList"]).info()

    section("6. 每列唯一值个数")
    for name, df in (("ratings", ratings), ("movies", movies), ("users", users)):
        print(f"\n[{name}]")
        # GenreList 是 list 列（不可哈希），改为统计题材标签数
        if "GenreList" in df.columns:
            print(f"GenreList     {len(genres)} 个不同题材标签（按 | 拆分后）")
        print(df.drop(columns=["GenreList"], errors="ignore").nunique().to_string())

    section("7. 数据类型")
    for name, df in (("ratings", ratings), ("movies", movies), ("users", users)):
        print(f"\n[{name}]")
        print(df.dtypes.astype(str).to_string())

    return comparison


if __name__ == "__main__":
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "dataset_info.txt"

    buf = io.StringIO()
    with redirect_stdout(buf):
        main()

    text = buf.getvalue()
    print(text)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n[已写入 {out_path.relative_to(config.ROOT)}]")
