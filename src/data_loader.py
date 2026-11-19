"""读取 ML-1M 原始 .dat 文件。

三个文件均为 `::` 分隔、无表头、latin-1 编码的纯文本：

    ratings.dat  UserID::MovieID::Rating::Timestamp
    movies.dat   MovieID::Title::Genres
    users.dat    UserID::Gender::Age::Occupation::Zip-code

字段定义见 ml-1m/README。本模块只做读取与最小化解析（提年份、拆题材），
不做任何过滤、编码或统计，避免后续划分/特征阶段出现隐式泄漏。
"""

from __future__ import annotations

import pandas as pd

from .config import DATA_DIR, MOVIES_PATH, RATINGS_PATH, USERS_PATH

RATINGS_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_COLUMNS = ["MovieID", "Title", "Genres"]
USERS_COLUMNS = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]


def _read_dat(path, names):
    """读取 `::` 分隔的 .dat 文件，缺失时给出可操作的报错。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            f"请从 https://files.grouplens.org/datasets/movielens/1m/ 下载并解压，"
            f"使 ratings.dat / movies.dat / users.dat 位于 {DATA_DIR} 下。"
        )
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=names,
        encoding="latin-1",
    )


def load_ratings(path=RATINGS_PATH) -> pd.DataFrame:
    """返回评分表：UserID, MovieID, Rating, Timestamp。"""
    df = _read_dat(path, RATINGS_COLUMNS)
    return df.astype(
        {
            "UserID": "int32",
            "MovieID": "int32",
            "Rating": "int8",
            "Timestamp": "int64",
        }
    )


def load_movies(path=MOVIES_PATH) -> pd.DataFrame:
    """返回电影表，并附带解析出的 Year 与去括号的 TitleClean。

    ML-1M 的标题形如 ``Toy Story (1995)``；少数条目无年份，
    此时 Year 为缺失值（NaT/NaN），不丢弃该行。
    """
    df = _read_dat(path, MOVIES_COLUMNS)
    df = df.astype({"MovieID": "int32", "Title": "string", "Genres": "string"})

    year = df["Title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    df["Year"] = pd.to_numeric(year, errors="coerce").astype("Int16")
    df["TitleClean"] = (
        df["Title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    )

    # 题材拆分为列表，便于后续 multi-hot 编码与统计
    df["GenreList"] = df["Genres"].str.split("|")
    return df


def load_users(path=USERS_PATH) -> pd.DataFrame:
    """返回用户表：UserID, Gender, Age, Occupation, Zip-code。"""
    df = _read_dat(path, USERS_COLUMNS)
    return df.astype(
        {
            "UserID": "int32",
            "Gender": "category",
            "Age": "int8",
            "Occupation": "int8",
            "Zip-code": "string",
        }
    )


def load_all():
    """一次性读取三张表，返回 (ratings, movies, users)。"""
    return load_ratings(), load_movies(), load_users()


if __name__ == "__main__":
    ratings, movies, users = load_all()
    print(f"ratings {ratings.shape}  movies {movies.shape}  users {users.shape}")
    print(ratings.dtypes.to_string())
    print(movies[["MovieID", "Title", "TitleClean", "Year", "Genres"]].head(3).to_string())
    print(users.head(3).to_string())
