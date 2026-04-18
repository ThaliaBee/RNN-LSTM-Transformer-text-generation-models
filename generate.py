"""
generate.py
自回归文本生成脚本，支持 RNN / LSTM / Transformer。

用法示例：
  python generate.py --model rnn --prompt "今天"
  python generate.py --model lstm --prompt "北京时间" --length 200
  python generate.py --model transformer --prompt "中国队" --temperature 0.8

参数说明：
  --model       : rnn / lstm / transformer
  --checkpoint  : 模型文件路径（默认 checkpoints/<model>.pt）
  --data_dir    : 词表所在目录（默认 data/processed）
  --prompt      : 输入提示文本（几个汉字即可）
  --length      : 生成字符数（默认 200）
  --temperature : 采样温度，越低越保守，越高越随机（默认 1.0）
"""

import os
import argparse

import torch

from dataset import load_vocab
from model import build_model


def encode(text: str, char2id: dict, unk_id: int = 1) -> list:
    """将字符串转为 ID 列表，未知字符用 <unk> 代替。"""
    return [char2id.get(ch, unk_id) for ch in text]


def decode(ids: list, id2char: dict) -> str:
    """将 ID 列表转回字符串。"""
    return ''.join(id2char.get(i, '?') for i in ids)


def main():
    parser = argparse.ArgumentParser(description='字符级语言模型文本生成')
    parser.add_argument('--model',       default='rnn',
                        choices=['rnn', 'lstm', 'transformer'],
                        help='模型类型')
    parser.add_argument('--checkpoint',  default=None,
                        help='checkpoint 路径（默认 checkpoints/<model>.pt）')
    parser.add_argument('--data_dir',    default='data/processed',
                        help='词表目录')
    parser.add_argument('--prompt',      default='今天',
                        help='输入提示文本')
    parser.add_argument('--length',      type=int,   default=200,
                        help='生成字符数')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度（0~2，推荐 0.8~1.2）')
    args = parser.parse_args()

    # ── 设备 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 词表 ──
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    char2id, id2char = load_vocab(vocab_path)
    vocab_size = len(char2id)

    # ── 加载模型 ──
    ckpt_path = args.checkpoint or os.path.join('checkpoints', f'{args.model}.pt')
    if not os.path.exists(ckpt_path):
        print(f"错误：找不到模型文件 {ckpt_path}")
        print(f"请先运行：python train.py --model {args.model}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(args.model, vocab_size).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print(f"模型: {args.model.upper()}  "
          f"（训练到第 {ckpt['epoch']} epoch，"
          f"验证 loss={ckpt['valid_loss']:.4f}）")
    print(f"温度: {args.temperature}  生成长度: {args.length}")
    print("-" * 50)

    # ── 编码 prompt ──
    prompt_ids = encode(args.prompt, char2id)
    if not prompt_ids:
        print("prompt 为空，请输入至少一个字符。")
        return

    # ── 生成 ──
    generated_ids = model.generate(
        prompt_ids,
        max_new=args.length,
        temperature=args.temperature,
        device=device,
    )

    # ── 输出（prompt 加粗区分，纯文本则用括号标注） ──
    prompt_text    = decode(prompt_ids, id2char)
    generated_text = decode(generated_ids[len(prompt_ids):], id2char)

    print(f"[提示] {prompt_text}")
    print(f"[生成]\n{generated_text}")


if __name__ == '__main__':
    main()
