# Spam Email Classification Using Multiple Machine Learning Models

课程级文本二分类项目：在相同训练/测试划分下，对比 **Logistic Regression**、**KNN**、**Multinomial Naive Bayes**、**MLP** 与 **LSTM** 在垃圾邮件识别上的表现，并输出指标、混淆矩阵与对比图。

## 目录结构

```
spam_email_classification/
├── data/
│   ├── raw/              # 原始 CSV（示例见 emails.csv）
│   └── processed/        # 预处理后数据（运行后生成）
├── notebooks/            # 可选：实验性 Jupyter 笔记
├── src/
│   ├── __init__.py
│   ├── config.py         # 路径与超参数
│   ├── preprocess.py     # 文本清洗与可选停用词
│   ├── features.py       # BoW / TF-IDF；LSTM 词表与 padding
│   ├── ml_models.py      # sklearn 模型构造
│   ├── dl_models.py      # PyTorch MLP / LSTM
│   ├── dataset.py        # torch Dataset 封装
│   ├── train_ml.py       # 传统 ML 训练与持久化
│   ├── train_dl.py       # 深度学习训练与最佳权重
│   ├── evaluate.py       # 指标、混淆矩阵、柱状对比图
│   └── utils.py          # IO、标签编码、目录创建
├── models/               # 训练得到的 .pkl / .pt
├── outputs/
│   ├── figures/          # 混淆矩阵与对比图 PNG
│   ├── metrics/          # JSON 指标
│   └── predictions/      # 测试集预测 CSV
├── main.py               # 统一入口
├── requirements.txt
└── README.md
```

## 环境安装

要求 **Python 3.10+**。建议使用 **Python 3.12**（或 3.10/3.11）创建虚拟环境：部分依赖在 **Python 3.14** 上可能尚未完全稳定，容易出现安装耗时或运行时异常。

```bash
cd spam_email_classification
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

若启用 **停用词**（在 `src/config.py` 中设置 `REMOVE_STOPWORDS = True`），首次运行 NLTK 会自动下载 `stopwords` 语料。

### macOS 上 “Python 意外退出” / Matplotlib 卡住

项目已在 `main.py` 中设置 `MPLCONFIGDIR` 指向项目内目录，并在 `src/evaluate.py` 中强制使用非交互后端 **`Agg`**（`matplotlib.use("Agg")`），避免默认 GUI 后端在部分系统上与字体缓存冲突导致进程崩溃。若仍异常，可尝试用 **Python 3.12** 重新建虚拟环境后再安装依赖。

## 数据集格式

CSV 至少包含两列（列名可在 `src/config.py` 中修改，默认为 `text` 与 `label`）：

| 列名   | 说明 |
|--------|------|
| `text` | 邮件正文或主题等原始文本 |
| `label`| `spam` / `ham`，或 `1` / `0`（也支持 `yes/no`、`true/false` 等别名） |

将文件放在 `data/raw/`，或通过 `--data` 指定路径。项目自带小型示例 `data/raw/emails.csv` 便于直接跑通流程。

### 使用 Kaggle 数据集

支持三种获取方式，**任选其一**。

**方式 0 —— `kagglehub`（推荐，免登录配置）：**

```bash
pip install kagglehub
python scripts/download_kaggle.py \
    --slug abdmental01/email-spam-dedection \
    --output data/raw/email_spam.csv
```

`scripts/download_kaggle.py` 会：

1. 通过 `kagglehub` 下载到本地缓存（默认 `~/.cache/kagglehub/...`）
2. 自动在解压目录里找 CSV（多 CSV 会列出来，可用 `--csv-name` 指定）
3. 调用 `prepare_kaggle.py` 把列名归一化为 `text,label`，写到 `data/raw/email_spam.csv`

**方式 1 —— Kaggle 官方 CLI（已配置 token）：**

```bash
pip install kaggle
kaggle datasets download -d uciml/sms-spam-collection-dataset -p data/raw --unzip
```

**方式 2 —— 网页下载**：直接从 Kaggle 数据集页面 `Download` 后解压，把 CSV 放到 `data/raw/`。

下载后再二选一：

A. **转成项目标准格式**（推荐，便于复用）：

   ```bash
   python scripts/prepare_kaggle.py \
       --input data/raw/spam.csv \
       --output data/raw/sms_spam.csv \
       --encoding latin-1 \
       --dedup
   ```

   脚本会：自动识别 `v1/v2`、`Category/Message` 等常见列名 → 输出含 `text,label` 的标准 CSV，并打印类别分布。

   然后照常训练：

   ```bash
   python main.py --mode all --feature tfidf --runs 3 \
       --data data/raw/sms_spam.csv
   ```

B. **不转换，直接用原始列名**（适合一次性试跑）：

   ```bash
   python main.py --mode ml --feature tfidf \
       --data data/raw/spam.csv \
       --text-col v2 --label-col v1 \
       --encoding latin-1
   ```

   `--text-col` / `--label-col` / `--encoding` 让 `main.py` 直接吃任意列名/编码的 CSV。

## 运行示例

在项目根目录（`spam_email_classification/`）执行：

```bash
# 仅传统机器学习（LR / KNN / MultinomialNB）
python main.py --mode ml --feature tfidf

# 仅深度学习（MLP + LSTM）
python main.py --mode dl --feature tfidf

# 全部模型
python main.py --mode all --feature bow
```

常用参数：

- `--mode`：`ml` | `dl` | `all`
- `--feature`：`bow` | `tfidf`（作用于 sklearn 与 MLP；LSTM 使用独立词表）
- `--data`：自定义 CSV 路径（默认 `data/raw/emails.csv`）
- `--seed`：基础随机种子（默认 42）
- `--runs N`：用 `seed, seed+1, ...` 重复 N 次实验，自动汇总每个模型 **mean ± std**
- `--epochs / --batch-size / --lr`：覆写 DL 的训练超参（不传则用 `config.py`）
- `--text-col / --label-col`：CSV 列名（默认 `text` / `label`，便于直接吃 Kaggle 原文件）
- `--encoding`：CSV 编码（Kaggle SMS Spam Collection 用 `latin-1`）
- `--no-class-weight`：关闭 DL 的 class-balanced loss（默认开启，用于消融实验）
- `--tag NAME`：把所有产物落到 `outputs/<NAME>/` 与 `models/<NAME>/`，避免不同实验互相覆盖；`--tag auto` 用时间戳自动命名
- `--log-level`：`DEBUG` / `INFO` / `WARNING` / `ERROR`

### 实验报告推荐命令

不同实验放在不同 `--tag` 子目录下，互不覆盖，方便最后一起汇总：

```bash
# 1) 基线：全部模型 + TF-IDF + 类权重平衡
python main.py --mode all --feature tfidf --runs 3 --epochs 15 \
    --data data/raw/email_spam.csv \
    --tag baseline

# 2) 消融：关掉类权重，对照 spam 不平衡的影响
python main.py --mode dl --feature tfidf --runs 3 --epochs 15 \
    --data data/raw/email_spam.csv \
    --no-class-weight --tag no_class_weight

# 3) 特征对比：BoW vs TF-IDF
python main.py --mode all --feature bow --runs 3 --epochs 15 \
    --data data/raw/email_spam.csv \
    --tag bow

# 4) 只快速跑传统 ML（秒级，便于反复试参数）
python main.py --mode ml --feature tfidf --runs 5 --seed 42 --tag ml_quick
```

不传 `--tag` 时维持原扁平结构（`outputs/figures/...`、`models/...`），适合「最后一次性出报告」的场景；但要注意每次都会**覆盖**前一次的同名产物。

### 类不平衡处理

`abdmental01/email-spam-dedection` 这类数据集中 ham ≈ 88%、spam ≈ 12%。原始 LSTM 在这种分布下，未经处理会陷入「全部预测 ham」的局部解（验证 loss 卡在 ≈ 0.36，正好等于先验交叉熵），导致 spam recall ≈ 0。

项目通过三个机制解决（默认全部开启）：

1. **Class-balanced cross-entropy**（`config.USE_CLASS_WEIGHT = True`）：按 `N / (K · n_c)` 给每个类别加权，等价于 sklearn 的 `class_weight='balanced'`。可用 `--no-class-weight` 关闭以做消融。
2. **梯度裁剪**（`config.GRAD_CLIP_NORM = 5.0`）：`clip_grad_norm_` 防止 LSTM 梯度爆炸。
3. **双向 LSTM**（`config.LSTM_BIDIRECTIONAL = True`）：拼接末态前/后向隐藏向量，对短文本通常 +1～3% F1。

实测在 `email_spam.csv` 上，启用后 LSTM 的 test F1 从 ≈ 0 提升到 0.80+，spam recall ≈ 0.86。

## 输出说明

下表中 `<tag>/` 是可选的 `--tag` 子目录；不传 `--tag` 时所有产物落在扁平的 `outputs/`、`models/` 下。

| 路径 | 内容 |
|------|------|
| `data/processed/processed.csv` | 原始列 + `clean_text` |
| `models/<tag>/vectorizer_{bow|tfidf}.pkl` | 文本向量器 |
| `models/<tag>/{logistic_regression,knn,naive_bayes}_{feature}.pkl` | sklearn 模型 |
| `models/<tag>/mlp_{feature}.pt` | MLP 检查点 |
| `models/<tag>/lstm.pt` | LSTM 检查点（含词表） |
| `outputs/<tag>/metrics/metrics_{mode}_{feature}_run*_seed*.json` | **每次 run** 的逐模型指标 |
| `outputs/<tag>/metrics/summary_{mode}_{feature}.csv` | 汇总表（多 run 时自动给出 `*_mean / *_std`，可直接贴报告） |
| `outputs/<tag>/metrics/summary_{mode}_{feature}.json` | 同上的 JSON 版 |
| `outputs/<tag>/metrics/label_mapping.json` | 标签到 0/1 的映射 |
| `outputs/<tag>/figures/cm_*.png` | 各模型混淆矩阵 |
| `outputs/<tag>/figures/curves_{mlp,lstm}_*.png` | **训练/验证 loss 曲线** |
| `outputs/<tag>/figures/model_comparison_*.png` | 多模型指标柱状对比 |
| `outputs/<tag>/predictions/*.csv` | 测试集预测结果 |

