# 中文字符级语言模型

基于中文新闻语料，对比 RNN、LSTM、Transformer 三种架构在文本生成任务上的效果。输入几个汉字，模型自动续写后续内容。

模型已训练完毕，下载后可直接运行生成脚本。

---

## 项目结构

```
├── preprocess.py          # 数据预处理（清洗 + 构建词表 + 数值化）
├── dataset.py             # PyTorch Dataset 定义
├── model.py               # RNN / LSTM / Transformer 模型定义
├── train.py               # 统一训练脚本（支持早停）
├── generate.py            # 文本生成脚本
├── checkpoints/
│   ├── rnn.pt             # 训练好的 RNN 模型
│   ├── lstm.pt            # 训练好的 LSTM 模型
│   └── transformer.pt     # 训练好的 Transformer 模型
└── data/processed/
    ├── corpus.txt         # 清洗后的纯文本语料
    ├── vocab.json         # 字符级词表
    ├── train_ids.npy      # 训练集
    ├── valid_ids.npy      # 验证集
    └── test_ids.npy       # 测试集
```

---

## 环境要求

- Python 3.10
- torch 2.5.1+cu124（需要 CUDA 12.4+，纯 CPU 环境见下方说明）
- numpy 1.26.4
- tqdm 4.67.3

**安装依赖：**

```bash
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4 tqdm==4.67.3
```

**仅使用 CPU（无 GPU）：**

```bash
pip install torch==2.5.1 numpy==1.26.4 tqdm==4.67.3
```

> CPU 模式下生成速度正常，训练速度较慢。

---

## 文本生成（直接使用）

```bash
# RNN 模型生成
python generate.py --model rnn --prompt "今天"

# LSTM 模型生成
python generate.py --model lstm --prompt "北京时间"

# Transformer 模型生成
python generate.py --model transformer --prompt "中国队"
```

**常用参数：**

| 参数            | 说明                                     | 默认值 |
| --------------- | ---------------------------------------- | ------ |
| `--model`       | 模型类型：`rnn` / `lstm` / `transformer` | `rnn`  |
| `--prompt`      | 输入提示文本                             | `今天` |
| `--length`      | 生成字符数                               | `200`  |
| `--temperature` | 采样温度，越低越保守，越高越随机         | `1.0`  |

**示例：**

```bash
# 生成 300 字，温度 0.8（较流畅）
python generate.py --model lstm --prompt "北京时间" --length 300 --temperature 0.8

# 贪心解码（最保守）
python generate.py --model transformer --prompt "中国队" --temperature 0.0

# 三个模型对比同一个 prompt
for model in rnn lstm transformer; do
    echo "===== $model ====="
    python generate.py --model $model --prompt "今天的比赛" --length 150 --temperature 0.8
done
```

---

## 从头训练（可选）

如需用自己的数据重新训练，按以下步骤操作。

**第一步：准备数据**

将 `.txt` 格式的新闻文本放入 `news_text/` 目录，每篇文章一个文件，然后运行预处理：

```bash
python preprocess.py --input news_text --output_dir data/processed
```

**第二步：训练模型**

```bash
python train.py --model rnn
python train.py --model lstm
python train.py --model transformer
```

**训练参数：**

| 参数           | 说明     | 默认值 |
| -------------- | -------- | ------ |
| `--model`      | 模型类型 | `rnn`  |
| `--seq_len`    | 序列长度 | `128`  |
| `--batch_size` | 批大小   | `64`   |
| `--lr`         | 学习率   | `1e-3` |
| `--patience`   | 早停轮数 | `5`    |

训练采用早停策略，验证集 loss 连续 `patience` 个 epoch 不下降时自动停止，最优模型保存至 `checkpoints/`。

