# MovieLens 1M 电影推荐系统

基于 MovieLens 1M 数据集的电影推荐系统课程项目。

项目目标是实现一个基础的推荐函数：**给定一名用户，推荐至少 5 部该用户尚未评分过的电影**，并在统一的用户集与候选集上对比 ItemCF、BPR、LRURec 三种算法的推荐结果。

---

## 1. 数据集

### 1.1 下载

MovieLens 1M 由 GroupLens Research 发布，下载地址：

<https://files.grouplens.org/datasets/movielens/1m/>

### 1.2 目录要求

将解压后的 `ml-1m/` 目录放在**项目根目录**下，最终路径必须为：

```
movie-recommender/
└── ml-1m/
    ├── ratings.dat
    ├── movies.dat
    └── users.dat
```

数据读取代码按上述路径读取。

### 1.3 文件格式

三个文件均为 `::` 分隔的纯文本（无表头，编码为 latin-1），由 `ml-1m/README` 定义：

| 文件 | 字段 | 说明 |
|---|---|---|
| `ratings.dat` | `UserID::MovieID::Rating::Timestamp` | 评分 1–5 的整数（整星），`Timestamp` 为 Unix 秒 |
| `movies.dat` | `MovieID::Title::Genres` | `Genres` 以 `\|` 分隔，可多值（如 `Animation\|Children's\|Comedy`） |
| `users.dat` | `UserID::Gender::Age::Occupation::Zip-code` | `Gender` 为 `M`/`F`；`Age`、`Occupation` 为分组编号 |

### 1.4 规模

- 评分：**1,000,209** 条，取值 1–5 的整数
- 用户：**6,040** 位（`UserID` 1–6040）
- 电影：`movies.dat` 中 **3,883** 部，评分实际覆盖其中 **3,706** 部（`MovieID` 范围 1–3952，部分 ID 无对应影片）
- 稀疏度：评分矩阵 6,040 × 3,706，仅 **4.47%** 的位置有值
- 每位用户至少有 20 条评分

统计口径：评分数取 `ratings.dat` 行数；用户数、电影数取 `ratings.dat` 中出现的唯一 ID 数；密度 = 评分数 ÷（用户数 × 电影数）。与 `ml-1m/README` 的官方描述（约 3,900 部电影、6,040 位用户、100 万条评分）一致。

### 1.5 版本控制说明

`ml-1m/` 与所有 `.zip`、`.pdf` 均已被 [.gitignore](.gitignore) 排除。

**本仓库不包含任何原始数据。** 克隆仓库后必须按 1.1、1.2 自行下载并解压，否则代码无法运行。

### 1.6 引用

使用本数据集须引用：

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: <http://dx.doi.org/10.1145/2827872>

### 1.7 使用声明

数据仅用于课程学习目的：

- 不再分发原始数据（本仓库已通过 `.gitignore` 排除）
- 不用于商业或其他营利用途
- 不暗示明尼苏达大学或 GroupLens Research Group 对本项目的认可

完整条款见 `ml-1m/README` 的 `USAGE LICENSE` 一节。

---

## 2. 算法

三种算法分别对应课程要求的三类来源，且各自承担不同的对照角色：

| 算法 | 角色 | 定位与核心思想 |
|---|---|---|
| **ItemCF** | 课程算法（Lecture 7） | **基于物品共现相似度的邻域推荐**。用用户–物品评分矩阵计算 item–item 相似度（Adjusted cosine），对候选物品按相似度加权聚合邻居评分。无训练过程（仅需预计算相似度矩阵），结果可直接解释为"因为你喜欢 X，所以推荐与 X 相似的 Y" |
| **BPR** | 参考文献复现 | **贝叶斯个性化排序**（Bayesian Personalized Ranking）。隐因子模型搭配**成对（pairwise）排序目标**，用 (用户, 正样本, 负样本) 三元组优化，直接针对排序质量而非拟合评分值，本质是隐式反馈模型 |
| **LRURec** | 2024 新算法 | **线性递归单元序列推荐**（WSDM 2024）。把用户历史按时间排成行为序列，用线性递归单元（Linear Recurrent Unit）建模序列依赖，预测下一次交互，显式利用**顺序信息**——这是 ItemCF 与 BPR 都不建模的维度 |

三者构成递进关系：邻域统计（无参数学习）→ 隐因子 + 排序目标（矩阵分解范式）→ 序列神经网络（时序建模），便于在报告中对比"方法复杂度"与"指标收益"是否成正比。

参考来源：

- ItemCF：课程 Lecture 7《Information Retrieval and Recommendations》，Item/User based collaborative filtering 一节（相似度指标含 Cosine、Adjusted cosine、PCC 等）
- BPR：Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009. <https://arxiv.org/pdf/1205.2618>
- LRURec：*LRURec: Rethinking Recommender Systems for LRU Cache*, WSDM 2024. <https://doi.org/10.1145/3616855.3635760>（官方实现：<https://github.com/yueqirex/LRURec>）

### 2.1 实现状态与当前结果

| 算法 | 状态 | 代码 |
|---|---|---|
| ItemCF | 已实现，含 48 组网格搜索 | `src/models/itemcf.py` |
| BPR | 待实现 | — |
| LRURec | 待实现 | — |

ItemCF 在验证集（6,040 位用户）上选出的最优配置为
`weighting=binary, alpha=0, k_neighbors=20`，48 组网格的完整记录见
`outputs/results/itemcf_tuning.txt`。测试集结果：

| 模型 | 训练数据 | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| 流行度基线 | train | 0.0351 | 0.0174 | 0.0122 |
| ItemCF | train | 0.0424 | 0.0201 | 0.0135 |
| ItemCF | train + valid | 0.0452 | 0.0215 | 0.0143 |

口径：6,040 位用户全部计入；每用户留出 1 条，在全部 3,704 部候选电影上排序，
训练集已交互的电影置为不可推荐；留出评分 1–3 星按未命中计；物品冷启动
（test 中有 2 部电影未在 train 出现）计入分母并按未命中处理。

- 相对流行度基线提升约 **21%**（0.0351 → 0.0424），说明物品共现相似度带来了真实的个性化增益。
- 若只评测"留出评分 ≥ 4"的 3,513 位用户，Recall@10 为 0.0544（train）/ 0.0572（train+valid）。
- train+valid 重训比仅用 train 高约 6.6%（NDCG 0.0201 → 0.0215）。
- 消融：`drop_negative`（负相似度置零）在最优配置下对指标无影响——`k=20` 时邻居全为正相似度。

---

## 3. 环境与运行

### 3.1 环境

实测环境：**Python 3.12.16**，conda 环境 `recsys`，Windows 22H2，无 GPU。

- `requirements.txt`：直接依赖，版本为实测版本
- `requirements.lock.txt`：`pip freeze` 全量快照，含所有传递依赖，用于精确复现

```bash
# 1) 创建并激活环境
conda create -n recsys python=3.12 -y
conda activate recsys

# 2) 安装直接依赖
pip install -r requirements.txt

# 3) 单独安装 CPU 版 torch
#    本机无 NVIDIA GPU；PyPI 默认 wheel 自带 CUDA 运行时（2~3 GB），
#    因此走 CPU 索引，实测版本 2.15.0+cpu
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

> 在未执行过 `conda init` 的 PowerShell 中 `conda activate` 可能不可用。
> 此时直接调用解释器绝对路径即可，例如：
> `& "$env:USERPROFILE\.conda\envs\recsys\python.exe" -m src.data_loader`

### 3.2 数据准备与校验

按 1.2 节放置 `ml-1m/` 后，先做读取自检（约 5 秒），预期输出
`ratings (1000209, 4)  movies (3883, 6)  users (6040, 5)`：

```bash
python -m src.data_loader
```

再生成评分项 "Dataset information" 要求的七项信息，同时写入
`outputs/results/dataset_info.txt`：

```bash
python -m src.dataset_info
```

### 3.3 数据划分

每用户按时间戳时序留一（最后 1 条 → test，倒数第 2 条 → valid，其余 → train），
附带防泄漏断言（三集互不重叠、每用户名额正确、时间序单调），结果保存为 parquet：

```bash
python -m src.split
```

产物：`outputs/splits/{train,valid,test}.parquet`（三个模型共用同一份划分）、
`outputs/results/split_summary.txt`。

### 3.4 探索性分析

```bash
python -m src.eda
```

产物：`outputs/figures/01_rating_distribution.png`、`02_movie_long_tail.png`、
`03_user_activity.png`、`04_genre_popularity.png`，统计摘要写入
`outputs/results/eda_stats.txt`。

### 3.5 模型训练与评估

统一评估层 `src/evaluate.py`：全目录排序（3,704 部候选）、每用户 1 条留出、
屏蔽训练集已交互，输出 Recall@K / NDCG@K / MRR@K（`K = 10`，见 `src/config.py`）。

| 命令 | 内容 |
|---|---|
| `python -m src.models.popularity` | 流行度基线（下界参照，同时用于自检评估流程） |
| `python -m src.models.itemcf` | ItemCF：48 组网格搜索 → 测试集评估 → 负相似度置零 A/B → train+valid 重训 |

ItemCF 的网格为 `weighting ∈ {binary, rating, centered}` ×
`alpha ∈ {0, 0.25, 0.5, 1.0}` × `k_neighbors ∈ {20, 50, 100, 200}`，
最优配置与完整表格写入 `outputs/results/itemcf_tuning.txt`（CPU 单次运行数分钟）。

BPR 与 LRURec 的命令随对应模块实现后补充。

### 3.6 交互界面

`streamlit` 已列入依赖，界面脚本随最终交付补充。

---

## 4. 项目结构

```text
movie-recommender/
├── ml-1m/                        原始数据（已 .gitignore，需自行下载）
├── src/
│   ├── config.py                 路径、随机种子、划分与评估参数（所有路径以此为基准）
│   ├── data_loader.py            读取三个 .dat（`::` 分隔、latin-1），解析年份与题材
│   ├── dataset_info.py           评分项 Dataset information（七项信息）
│   ├── preprocess.py             用户画像编码、题材 multi-hot、隐式反馈二值化
│   ├── split.py                  每用户时序留一 + 防泄漏断言
│   ├── eda.py                    评分项 Data exploration（四张图表 + 统计摘要）
│   ├── evaluate.py               统一评估：TrainIndex 缓存 + Recall/NDCG/MRR
│   └── models/
│       ├── __init__.py
│       ├── popularity.py         流行度基线
│       └── itemcf.py             评分项 Algorithms：ItemCF + 网格搜索
├── outputs/                      运行产物（已 .gitignore，可随时重建）
│   ├── splits/                   train/valid/test.parquet
│   ├── results/                  dataset_info / split_summary / eda_stats / itemcf_tuning
│   └── figures/                  EDA 图表
├── models/                       模型权重（BPR / LRURec 实现后写入，已 .gitignore）
├── requirements.txt              直接依赖
├── requirements.lock.txt         全量依赖快照
└── README.md
```

推荐运行顺序：`data_loader` → `dataset_info` / `eda` / `split` → `models.popularity` / `models.itemcf`。

`src/models/*` 优先复用 `outputs/splits/*.parquet`，文件不存在时回退到现场划分，
因此直接运行任一模型脚本也能自动就绪。
