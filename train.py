"""
train.py
统一训练脚本，支持 RNN / LSTM / Transformer 三种模型。

用法示例：
  python train.py --model rnn
  python train.py --model lstm --batch_size 128
  python train.py --model transformer --lr 3e-4

早停策略：验证集 loss 连续 patience 个 epoch 不下降则停止，
并自动保存最优 checkpoint 到 checkpoints/<model>.pt。
"""

import os
import json
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CharDataset, load_vocab
from model import build_model


# ──────────────────────────────────────────────
# 训练 / 验证单轮
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, total_tokens = 0.0, 0

    phase = 'train' if is_train else 'valid'
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc=phase, leave=False):
            x, y = x.to(device), y.to(device)           # (batch, seq_len)

            if is_train:
                optimizer.zero_grad()

            # 统一调用方式：RNN/LSTM 返回 (logits, hidden)，Transformer 只返回 logits
            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            # logits: (batch, seq_len, vocab_size) → reshape for CrossEntropy
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

            batch_tokens = y.numel()
            total_loss   += loss.item() * batch_tokens
            total_tokens += batch_tokens

    return total_loss / total_tokens


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='字符级语言模型训练')
    parser.add_argument('--model',      default='rnn',
                        choices=['rnn', 'lstm', 'transformer'],
                        help='模型类型')
    parser.add_argument('--data_dir',   default='data/processed', help='数据目录')
    parser.add_argument('--save_dir',   default='checkpoints',    help='模型保存目录')
    parser.add_argument('--seq_len',    type=int,   default=128,  help='序列长度')
    parser.add_argument('--batch_size', type=int,   default=64,   help='批大小')
    parser.add_argument('--lr',         type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience',   type=int,   default=5,    help='早停 patience')
    args = parser.parse_args()

    # ── 设备 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ── 加载词表和数据 ──
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    char2id, id2char = load_vocab(vocab_path)
    vocab_size = len(char2id)
    print(f"词表大小: {vocab_size}")

    train_ids = np.load(os.path.join(args.data_dir, 'train_ids.npy'))
    valid_ids = np.load(os.path.join(args.data_dir, 'valid_ids.npy'))

    train_set = CharDataset(train_ids, seq_len=args.seq_len)
    valid_set = CharDataset(valid_ids, seq_len=args.seq_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"训练样本: {len(train_set):,}  验证样本: {len(valid_set):,}")

    # ── 构建模型 ──
    model = build_model(args.model, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {args.model.upper()}  参数量: {n_params:,}")

    # ── 优化器 & 损失 ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # <pad>=0 的位置不计入 loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ── 保存目录 ──
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f'{args.model}.pt')

    # ── 早停状态 ──
    best_valid_loss = float('inf')
    patience_count  = 0
    epoch = 0

    print(f"\n开始训练（早停 patience={args.patience}）...\n")
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train PPL':>10}  "
          f"{'Valid Loss':>11}  {'Valid PPL':>10}  {'状态':>6}")
    print("-" * 68)

    while True:
        epoch += 1
        train_loss = run_epoch(model, train_loader, criterion, optimizer,
                               device, is_train=True)
        valid_loss = run_epoch(model, valid_loader, criterion, optimizer,
                               device, is_train=False)

        train_ppl = math.exp(min(train_loss, 20))   # 防止 exp 溢出
        valid_ppl = math.exp(min(valid_loss, 20))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_count  = 0
            torch.save({
                'epoch':      epoch,
                'model_type': args.model,
                'vocab_size': vocab_size,
                'state_dict': model.state_dict(),
                'valid_loss': valid_loss,
            }, ckpt_path)
            status = '✓ 保存'
        else:
            patience_count += 1
            status = f'等待 {patience_count}/{args.patience}'

        print(f"{epoch:>6}  {train_loss:>11.4f}  {train_ppl:>10.2f}  "
              f"{valid_loss:>11.4f}  {valid_ppl:>10.2f}  {status}")

        if patience_count >= args.patience:
            print(f"\n早停触发（连续 {args.patience} 个 epoch 验证 loss 未改善）")
            break

    print(f"\n训练结束，最优验证 loss: {best_valid_loss:.4f}  "
          f"（PPL: {math.exp(min(best_valid_loss, 20)):.2f}）")
    print(f"最优模型已保存至: {ckpt_path}")


if __name__ == '__main__':
    main()
