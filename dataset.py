"""
dataset.py
字符级语言模型 PyTorch Dataset

用法示例：
  from dataset import CharDataset, load_vocab
  import numpy as np

  char2id, id2char = load_vocab('data/processed/vocab.json')

  train_ids = np.load('data/processed/train_ids.npy')
  train_set  = CharDataset(train_ids, seq_len=128)

  from torch.utils.data import DataLoader
  loader = DataLoader(train_set, batch_size=64, shuffle=True)

  for x, y in loader:
      # x: (batch, seq_len)  输入字符 ID
      # y: (batch, seq_len)  标签（x 右移一位）
      ...
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


def load_vocab(vocab_path: str):
    """
    读取 vocab.json，返回 (char2id, id2char)。
    id2char 的键为 int 类型。
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    char2id = vocab['char2id']
    id2char  = {int(k): v for k, v in vocab['id2char'].items()}
    return char2id, id2char


class CharDataset(Dataset):
    """
    字符级语言模型数据集（next-character prediction）。

    将字符 ID 序列切成长度为 seq_len 的固定窗口：
      x[i] = ids[i : i + seq_len]
      y[i] = ids[i+1 : i+1 + seq_len]   （右移一位，即预测下一个字符）

    参数：
      ids     : np.ndarray (int32)，字符 ID 序列（来自 train/valid/test_ids.npy）
      seq_len : 每个样本的序列长度，默认 128
    """

    def __init__(self, ids: np.ndarray, seq_len: int = 128):
        # 保留能构成完整样本的部分（丢弃末尾不足一个窗口的余量）
        self.seq_len = seq_len
        # 最多可切出的样本数
        n_samples = (len(ids) - 1) // seq_len
        # 截断到整数倍，方便索引
        usable_len = n_samples * seq_len + 1
        self.ids = torch.from_numpy(ids[:usable_len].astype(np.int64))

    def __len__(self):
        # 每隔 seq_len 取一个样本（无重叠切分）
        return (len(self.ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.ids[start : start + self.seq_len]
        y = self.ids[start + 1 : start + 1 + self.seq_len]
        return x, y


# ──────────────────────────────────────────────
# 快速验证入口
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    vocab_path = 'data/processed/vocab.json'
    train_path = 'data/processed/train_ids.npy'

    char2id, id2char = load_vocab(vocab_path)
    print(f"词表大小: {len(char2id)}")

    ids = np.load(train_path)
    print(f"训练集 token 数: {len(ids):,}")

    dataset = CharDataset(ids, seq_len=128)
    print(f"训练集样本数（seq_len=128）: {len(dataset):,}")

    x, y = dataset[0]
    print(f"\n第 0 个样本:")
    print(f"  x shape: {x.shape}  dtype: {x.dtype}")
    print(f"  y shape: {y.shape}  dtype: {y.dtype}")
    print(f"  x（前20字符）: {''.join(id2char[i.item()] for i in x[:20])}")
    print(f"  y（前20字符）: {''.join(id2char[i.item()] for i in y[:20])}")
