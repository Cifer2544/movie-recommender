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

---

## 3. 环境与运行

```text
TODO
```

待环境确定后补齐：Python 版本、依赖清单与安装方式、数据准备校验、三模型的训练/推理命令、交互界面启动方式。所有命令以实际跑通为准，不预先写入未验证的内容。

---

## 4. 项目结构

```text
TODO
```

待代码骨架落地后补齐：目录树、各模块职责、产出物（模型权重、评估结果、图表）的输出位置。
