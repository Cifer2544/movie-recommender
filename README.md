# MovieLens 1M 电影推荐系统

基于 MovieLens 1M 数据集的电影推荐系统课程项目。

项目目标是实现一个基础的推荐函数：**给定一名用户，推荐至少 5 部该用户尚未评分过的电影**，并在统一的用户集与候选集上对比 ItemCF、BPR、LRURec 三种算法的推荐结果。界面为 Streamlit 应用，模型离线训练后落盘，界面只做加载与推理。

**交付模型在测试集上的结果**（train + valid 重训口径，完整对比见 4.2）：

| 模型 | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| 流行度基线 | 0.0351 | 0.0174 | 0.0122 |
| ItemCF | 0.0452 | 0.0215 | 0.0143 |
| BPR | 0.0679 | 0.0330 | 0.0226 |
| LRURec | 0.0831 | 0.0471 | 0.0363 |

**只想把界面跑起来**：跳到 [1.1 六步跑起来](#11-六步跑起来)，约 10 分钟（大部分时间在下载依赖）。想复现完整实验（数据校验、EDA、网格搜索）则从 2.1 开始。

---

## 1. 快速开始

### 1.1 六步跑起来

如果目标只是把界面跑起来（例如组员在自己的电脑上查看），**不需要**执行 2.4 与 2.5
的 EDA、网格搜索脚本——`src.persist` 会直接用验证集选出的最优超参训练三个交付模型。

```powershell
# 1) 取代码
git clone https://github.com/Cifer2544/movie-recommender.git
cd movie-recommender

# 2) 放数据：下载 ml-1m.zip 解压，使三个 .dat 位于项目根目录的 ml-1m/ 下
#    https://files.grouplens.org/datasets/movielens/1m.zip
#    目录要求见 3.2

# 3) 环境（Python 3.12，与 2.1 节一致）
conda create -n recsys python=3.12 -y
conda activate recsys
pip install -r requirements.txt

# 4) torch 必须走 CPU 索引
#    不加 --index-url 会拉 CUDA 版 wheel（2~3 GB）
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 5) 划分与训练落盘
python -m src.split      # 数秒，生成 outputs/splits/*.parquet
python -m src.persist    # train + valid 重训三个模型并写入 models/

# 6) 启动
streamlit run src/app.py
```

### 1.2 预期输出与验收

`src.persist` 的预期输出是三行 `miss`，括号内为实测耗时：ItemCF 约 1.2 s、
BPR 约 18.4 s、LRURec 约 98.1 s；同一份数据与超参再次运行会变为 `hit` 并跳过训练。

浏览器打开 <http://localhost:8501>，默认 ItemCF + UserID 1 的首条推荐是
`The Shawshank Redemption (1994)`。

### 1.3 五个容易卡住的点

- **Python 版本**：必须是 3.12。`torch==2.15.0` 的 CPU wheel 按该版本安装。
- **数据目录**：必须是 `ml-1m/` 且包含 `ratings.dat`。`persist` 会读取全部三张表
  并在缺 parquet 时现场划分；只有界面进程才仅依赖 `outputs/splits/`。
- **torch 安装方式**：`requirements.txt` 有意不包含 torch（见该文件末尾注释），
  必须单独用 CPU 索引安装。
- **命令分隔符**：PowerShell 不支持 `&&`，多条命令需分行或用 `;` 连接。
- **重复运行**：`models/` 已在本地存在时 persist 直接命中缓存；想强制重训就删除
  对应目录，或直接运行 `python -m src.persist lrurec` 只处理单个模型。

---

## 2. 环境与完整流程

以下为完整复现流程。只想运行界面时按 1.1 即可，不必执行 2.4 与 2.5。

### 2.1 环境

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

### 2.2 数据准备与校验

按 3.2 节放置 `ml-1m/` 后，先做读取自检（约 5 秒），预期输出
`ratings (1000209, 4)  movies (3883, 6)  users (6040, 5)`：

```bash
python -m src.data_loader
```

再生成评分项 "Dataset information" 要求的七项信息，同时写入
`outputs/results/dataset_info.txt`：

```bash
python -m src.dataset_info
```

### 2.3 数据划分

每用户按时间戳时序留一（最后 1 条 → test，倒数第 2 条 → valid，其余 → train），
附带防泄漏断言（三集互不重叠、每用户名额正确、时间序单调），结果保存为 parquet：

```bash
python -m src.split
```

产物：`outputs/splits/{train,valid,test}.parquet`（三个模型共用同一份划分）、
`outputs/results/split_summary.txt`。

### 2.4 探索性分析

```bash
python -m src.eda
```

产物：`outputs/figures/01_rating_distribution.png`、`02_movie_long_tail.png`、
`03_user_activity.png`、`04_genre_popularity.png`，统计摘要写入
`outputs/results/eda_stats.txt`。

### 2.5 模型训练与评估

统一评估层 `src/evaluate.py`：全目录排序（3,704 部候选）、每用户 1 条留出、
屏蔽训练集已交互，输出 Recall@K / NDCG@K / MRR@K（`K = 10`，见 `src/config.py`）。

| 命令 | 内容 |
|---|---|
| `python -m src.models.popularity` | 流行度基线（下界参照，同时用于自检评估流程） |
| `python -m src.models.itemcf` | ItemCF：48 组网格搜索 → 测试集评估 → 负相似度置零 A/B → train+valid 重训 |
| `python -m src.models.bpr` | BPR：8 组网格（含验证集早停）→ 测试集评估 → train+valid 重训 |
| `python -m src.models.lrurec` | LRURec：4 组网格（含验证集早停）→ 测试集评估 → train+valid 重训 |

ItemCF 的网格为 `weighting ∈ {binary, rating, centered}` ×
`alpha ∈ {0, 0.25, 0.5, 1.0}` × `k_neighbors ∈ {20, 50, 100, 200}`；
BPR 的网格为 `embed_dim ∈ {32, 64}` × `lr ∈ {1e-3, 5e-3}` × `reg ∈ {1e-5, 1e-4}`，
每 2 个 epoch 在验证集上评估一次 NDCG@10 并按 patience=3 早停；
LRURec 的网格为 `embed_dim ∈ {32, 64}` × `lr ∈ {1e-3, 5e-3}`，
同样每 2 个 epoch 评估、patience=3 早停。
三者的最优配置与完整表格分别写入 `outputs/results/itemcf_tuning.txt`、
`outputs/results/bpr_tuning.txt`、`outputs/results/lrurec_tuning.txt`。

首次运行单个模型脚本约 10–35 分钟（含网格搜索与 train+valid 重训），均为纯 CPU 计算。

### 2.6 模型持久化与交互界面

界面不能每次启动都重新训练（BPR / LRURec 单次要 1–2 分钟），因此先把训练产物
落盘，界面只做加载与推理：

```powershell
python -m src.persist        # 训练三个最终交付模型并落盘（首次约 2 分钟）
streamlit run src/app.py     # 启动界面
```

`src/persist.py` 用 **train + valid** 全量重训（与评测口径②一致），产物写入
`models/<模型>_<超参哈希>_<数据指纹>/`：

| 文件 | 内容 |
|---|---|
| `state.npz` | 权重与索引（torch 权重转 ndarray 存放，加载时再转回） |
| `*.sparse.npz` | 稀疏矩阵（ItemCF 的 R / B / C 与相似度，BPR 与 LRURec 的历史屏蔽矩阵） |
| `meta.json` | 超参、训练时间、best_epoch、数据规模、fit 耗时 |

缓存规则见 `src/cache.py`：目录名由"超参 JSON 哈希 + 训练数据摘要"拼成，不用
`*_latest` 这类命名，避免改了数据或超参却静默读到旧权重。同一份数据与超参重复
运行会打印 `hit` 并跳过训练（实测二次运行不到 2 秒），改超参或改数据会自然产生
新目录；只存 state 而不 pickle 模型对象，因此重构 `src/models/` 里的类也不影响
已有缓存。落盘体积实测 ItemCF 5.4 MB、BPR 2.5 MB、LRURec 3.0 MB，合计约 11 MB，
已被 `.gitignore` 的 `/models/` 规则排除，不随仓库分发。

界面功能：

- 侧边栏选择算法、UserID（或随机换一个）、Top-K，可开启"三模型并排对比"；
  UserID 不限制上下界，输入数据集中不存在的 ID（如 9999）不会被控件拦下，而是
  在页面顶部给出警告条，说明该 ID 不在 MovieLens-1M（1 – 6040）中
- 主区展示用户画像（性别 / 年龄段 / 职业 / 评分条数 / 平均分）、Top-K 推荐、
  可折叠的历史记录，以及该算法在 test 上的 Recall@10 / NDCG@10 / MRR@10
- 展示历史取 train + valid，与模型训练数据一致；推荐在此基础上再屏蔽 test 留出的
  1 条，保证推荐结果都在用户评分历史之外
- 个性化结果为空时（LRURec 对没有正反馈的用户、或不在训练数据中的 UserID）按训练集
  评分人数兜底，仍给出至少 5 部未看过的电影，并在页面上标注"得分即评分人数"
- 片名在显示层做了还原：`movies.dat` 有 807 条采用冠词后置的目录排序写法
  （如 `Shawshank Redemption, The (1994)`，目的是让 *The Matrix* 排在 M 而不是 T），
  界面显示为 `The Shawshank Redemption (1994)`；MovieID 匹配与指标仍用原始列，
  算法脚本的 stdout 输出保留数据集原样

**注意**：`streamlit` 的热重载只可靠地重载入口脚本。改动 `src/service.py`、
`src/cache.py` 这类被导入的模块后，需要停止进程并重新 `streamlit run src/app.py`，
否则页面仍在用内存里的旧模块。

---

## 3. 数据集

### 3.1 下载

MovieLens 1M 由 GroupLens Research 发布，下载地址：

<https://files.grouplens.org/datasets/movielens/1m/>

### 3.2 目录要求

将解压后的 `ml-1m/` 目录放在**项目根目录**下，最终路径必须为：

```
movie-recommender/
└── ml-1m/
    ├── ratings.dat
    ├── movies.dat
    └── users.dat
```

数据读取代码按上述路径读取。

### 3.3 文件格式

三个文件均为 `::` 分隔的纯文本（无表头，编码为 latin-1），由 `ml-1m/README` 定义：

| 文件 | 字段 | 说明 |
|---|---|---|
| `ratings.dat` | `UserID::MovieID::Rating::Timestamp` | 评分 1–5 的整数（整星），`Timestamp` 为 Unix 秒 |
| `movies.dat` | `MovieID::Title::Genres` | `Genres` 以 `\|` 分隔，可多值（如 `Animation\|Children's\|Comedy`） |
| `users.dat` | `UserID::Gender::Age::Occupation::Zip-code` | `Gender` 为 `M`/`F`；`Age`、`Occupation` 为分组编号 |

### 3.4 规模

- 评分：**1,000,209** 条，取值 1–5 的整数
- 用户：**6,040** 位（`UserID` 1–6040）
- 电影：`movies.dat` 中 **3,883** 部，评分实际覆盖其中 **3,706** 部（`MovieID` 范围 1–3952，部分 ID 无对应影片）
- 稀疏度：评分矩阵 6,040 × 3,706，仅 **4.47%** 的位置有值
- 每位用户至少有 20 条评分

统计口径：评分数取 `ratings.dat` 行数；用户数、电影数取 `ratings.dat` 中出现的唯一 ID 数；密度 = 评分数 ÷（用户数 × 电影数）。与 `ml-1m/README` 的官方描述（约 3,900 部电影、6,040 位用户、100 万条评分）一致。

### 3.5 版本控制说明

`ml-1m/` 与所有 `.zip`、`.pdf` 均已被 [.gitignore](.gitignore) 排除。

**本仓库不包含任何原始数据。** 克隆仓库后必须按 3.1、3.2 自行下载并解压，否则代码无法运行。

### 3.6 引用

使用本数据集须引用：

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: <http://dx.doi.org/10.1145/2827872>

### 3.7 使用声明

数据仅用于课程学习目的：

- 不再分发原始数据（本仓库已通过 `.gitignore` 排除）
- 不用于商业或其他营利用途
- 不暗示明尼苏达大学或 GroupLens Research Group 对本项目的认可

完整条款见 `ml-1m/README` 的 `USAGE LICENSE` 一节。

---

## 4. 算法与结果

### 4.1 三类方法与参考来源

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

### 4.2 实现状态与当前结果

| 算法 | 状态 | 代码 |
|---|---|---|
| ItemCF | 已实现（48 组网格搜索） | `src/models/itemcf.py` |
| BPR | 已实现（8 组网格 + 验证集早停） | `src/models/bpr.py` |
| LRURec | 已实现（4 组网格 + 验证集早停） | `src/models/lrurec.py` |

ItemCF 在验证集（6,040 位用户）上选出的最优配置为
`weighting=binary, alpha=0, k_neighbors=20`，48 组网格的完整记录见
`outputs/results/itemcf_tuning.txt`。测试集结果：

| 模型 | 训练数据 | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| 流行度基线 | train | 0.0351 | 0.0174 | 0.0122 |
| ItemCF | train | 0.0424 | 0.0201 | 0.0135 |
| ItemCF | train + valid | 0.0452 | 0.0215 | 0.0143 |
| BPR | train | 0.0632 | 0.0313 | 0.0217 |
| BPR | train + valid | 0.0679 | 0.0330 | 0.0226 |
| LRURec | train | 0.0667 | 0.0352 | 0.0255 |
| LRURec | train + valid | 0.0831 | 0.0471 | 0.0363 |

口径：6,040 位用户全部计入；每用户留出 1 条，在全部 3,704 部候选电影上排序，
训练集已交互的电影置为不可推荐；留出评分 1–3 星按未命中计；物品冷启动
（test 中有 2 部电影未在 train 出现）计入分母并按未命中处理。

- 相对流行度基线提升约 **21%**（0.0351 → 0.0424），说明物品共现相似度带来了真实的个性化增益。
- 若只评测"留出评分 ≥ 4"的 3,513 位用户，Recall@10 为 0.0544（train）/ 0.0572（train+valid）。
- train+valid 重训比仅用 train 高约 6.6%（NDCG 0.0201 → 0.0215）。
- 消融：`drop_negative`（负相似度置零）在最优配置下对指标无影响——`k=20` 时邻居全为正相似度。

BPR 在验证集上选出的最优配置为 `embed_dim=32, lr=5e-3, reg=1e-5`（早停于 epoch 32），
完整网格见 `outputs/results/bpr_tuning.txt`：

- 相对 ItemCF 提升约 **49%**（0.0424 → 0.0632；重训口径 0.0452 → 0.0679）。
  pairwise 损失直接优化"正例排在负例之前"，比 ItemCF 的启发式邻域打分更贴合 Top-N 目标。
- 8 组中 `embed_dim=64, lr=1e-3` 的两组 `best_epoch` 均撞到 `max_epochs=50`，
  说明该学习率尚未收敛；最优组合 `lr=5e-3` 在 epoch 32 收敛，不受该截断影响，
  但更大 `max_epochs` 或更细的网格仍有提升空间。
- 正则强度影响很小（`reg=1e-5` 的 NDCG@10 = 0.0339 对 `1e-4` 的 0.0320），
  说明 ML-1M 上过拟合不是主要矛盾。

**关于 "train + valid" 一行**：该口径是在把 valid 并入训练集后重训、再评 test，
此时没有独立验证集可用于早停，BPR 只能固定在验证集选出的轮数（epoch 32）训练。
数据量仅增加 0.6%，但训练轮数无法随数据自适应，理论上存在欠拟合风险；
本数据上重训后指标反而更高（0.0632 → 0.0679），风险未显化。
该行仅作参考，主要结论以 "train" 行为准。

LRURec 在验证集上选出的最优配置为 `embed_dim=64, lr=1e-3`（早停于 epoch 10），
`d_state` 固定等于 `embed_dim`、`n_blocks=2`、`max_len=50`，完整网格见
`outputs/results/lrurec_tuning.txt`：

- 相对 BPR 提升约 **12%**（0.0313 → 0.0352）；重训口径从 0.0330 升到 0.0471（+43%）。
  这是三者中唯一显式建模交互顺序的模型，增益来自序列信息本身。
- 学习率的影响大于嵌入维度：`lr=5e-3` 的两组都在 epoch 8 触顶（valid NDCG 0.031–0.034），
  而 `d=64, lr=1e-3` 训到 epoch 10 达到 0.0453。递推状态对步长敏感，
  过大的 lr 会更快进入平台期。
- 重训口径的提升幅度（+34%）远大于 BPR（+5%）：把 valid 并入后每个用户的历史多一条交互，
  而 LRURec 的打分完全由序列决定。该行同样仅作参考。
- 训练目标是 one-target-per-user（每用户仅对序列末位预测一次），约 6,040 个训练对，
  比 BPR 的 98 万三元组少两个数量级，训练信号稀疏是它在低学习率下波动较大的原因；
  trade-off 的量化说明见 `src/models/lrurec.py` 模块文档第 5 条。
- 零正例用户（train 中无 ≥ 4 星交互，共 2 位：3598、4486）没有可编码的历史，
  打分置为 `-inf`，评估中一律未命中，`top_k` 返回空列表。这是序列模型的固有限制。

---

## 5. 项目结构

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
│   ├── cache.py                  模型缓存：指纹命名 + state 存取
│   ├── persist.py                训练最终交付模型并落盘（python -m src.persist）
│   ├── service.py                界面服务层：加载缓存模型、取历史、拼推荐表
│   ├── app.py                    Streamlit 界面入口（streamlit run src/app.py）
│   └── models/
│       ├── __init__.py
│       ├── popularity.py         流行度基线
│       ├── itemcf.py             评分项 Algorithms：ItemCF + 网格搜索
│       ├── bpr.py                BPR：pairwise 损失 + 负采样 + 验证集早停
│       └── lrurec.py             LRURec：复数线性递推 + 末位 next-item 预测
├── outputs/                      运行产物（已 .gitignore，可随时重建）
│   ├── splits/                   train/valid/test.parquet
│   ├── results/                  dataset_info / split_summary / eda_stats / *_tuning
│   └── figures/                  EDA 图表
├── models/                       模型缓存（已 .gitignore，由 python -m src.persist 生成）
├── requirements.txt              直接依赖
├── requirements.lock.txt         全量依赖快照
└── README.md
```

推荐运行顺序：`data_loader` → `dataset_info` / `eda` / `split` →
`models.popularity` / `models.itemcf` / `models.bpr` / `models.lrurec` →
`persist` → `streamlit run src/app.py`。

只在组员本机查看界面、或不需要复现网格搜索时，按 1.1 的最短路径执行即可。

`src/models/*` 优先复用 `outputs/splits/*.parquet`，文件不存在时回退到现场划分，
因此直接运行任一模型脚本也能自动就绪。
