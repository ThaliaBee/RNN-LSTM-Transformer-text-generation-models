"""
preprocess.py
中文新闻语料预处理脚本

用法：
  # 处理全部文件
  python preprocess.py --input news_text --output_dir data/processed

  # 只处理前 N 个文件（用于预览效果）
  python preprocess.py --input news_text --output_dir data/processed --limit 10

  # 指定序列长度（影响 dataset.py 的切片，不影响本脚本）
  python preprocess.py --input news_text --output_dir data/processed --seq_len 128

输出：
  data/processed/corpus.txt     — 合并后的纯文本语料
  data/processed/vocab.json     — char2id / id2char 词表
  data/processed/train_ids.npy  — 训练集字符 ID 序列
  data/processed/valid_ids.npy  — 验证集字符 ID 序列
  data/processed/test_ids.npy   — 测试集字符 ID 序列
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1. 文本清洗
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    对单篇文章做轻量清洗：
    - 全角空格（U+3000）替换为普通空格
    - 去掉 [方括号导航标签]，如 [视频直播室] [技术统计]
    - 去掉 URL（含嵌套在圆括号内的形式，如 (http://...)）
    - 去掉独占一行的记者署名，如 (Lai) (徐铮) (新浪体育)
    - 合并行内多余空格为单个空格
    - 合并连续 3 行以上的空行为 1 个空行
    - 去掉文章末尾多余空行
    """
    # 全角空格 → 半角空格
    text = text.replace('\u3000', ' ')

    # 去掉嵌套在圆括号内的 URL，如 (http://t.sina.com.cn)
    text = re.sub(r'\(https?://[^\)]*\)', '', text)
    # 去掉裸 URL（http:// 或 www. 开头，以空白/行尾结束）
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # 去掉 [方括号导航标签]，如 [视频直播室] [图文直播室] [1] 等
    text = re.sub(r'\[[^\]]{0,40}\]', '', text)

    # 逐行处理：去掉每行首尾空白，过滤署名行，合并行内多余空格
    _signature = re.compile(r'^\([^)]{1,40}\)$')
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if _signature.match(line):          # 跳过署名行
            continue
        line = re.sub(r' {2,}', ' ', line)  # 行内多余空格合并为一个
        lines.append(line)

    # 合并连续空行（最多保留 1 个空行）
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == '':
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append('')
        else:
            blank_count = 0
            cleaned_lines.append(line)

    # 去掉首尾空行
    text = '\n'.join(cleaned_lines).strip()
    return text


def is_valid(text: str, min_chinese: int = 50) -> bool:
    """
    过滤过短或无有效中文内容的文章。
    min_chinese：最少汉字数，默认 50。
    """
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) >= min_chinese


# ──────────────────────────────────────────────
# 2. 读取文件 → 清洗 → 写入 corpus.txt
# ──────────────────────────────────────────────

def build_corpus(input_dir: str, output_path: str, limit: int = None, min_chinese: int = 50):
    """
    读取 input_dir 下所有 .txt 文件，清洗后合并写入 output_path。
    每篇文章之间用一个空行分隔。
    limit: 只处理前 limit 个文件（None 表示全部）。
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob('*.txt'))

    if limit is not None:
        files = files[:limit]

    print(f"共找到文件: {len(files)} 篇（处理 {'全部' if limit is None else limit} 篇）")

    kept, dropped = 0, 0
    with open(output_path, 'w', encoding='utf-8') as fout:
        for fp in tqdm(files, desc='清洗文章'):
            try:
                raw = fp.read_text(encoding='utf-8', errors='replace')
            except Exception as e:
                print(f"  跳过 {fp.name}：读取错误 {e}")
                continue

            cleaned = clean_text(raw)

            if not is_valid(cleaned, min_chinese):
                dropped += 1
                continue

            fout.write(cleaned)
            fout.write('\n\n')   # 文章间空一行
            kept += 1

    print(f"保留: {kept} 篇，过滤: {dropped} 篇")
    print(f"语料已写入: {output_path}")


# ──────────────────────────────────────────────
# 3. 构建字符级词表
# ──────────────────────────────────────────────

def build_vocab(corpus_path: str, vocab_path: str, min_freq: int = 1):
    """
    统计 corpus.txt 中的全部字符，构建 char2id / id2char，
    保存为 vocab.json。
    特殊符号：<pad>=0, <unk>=1
    min_freq：最低出现频次，默认 1（不过滤）。
    """
    print("统计字符频次...")
    counter = Counter()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='统计'):
            counter.update(line)

    # 按频次降序排列
    vocab_chars = [ch for ch, cnt in counter.most_common() if cnt >= min_freq]

    # 保留可见字符，去掉控制字符（保留换行 \n）
    vocab_chars = [ch for ch in vocab_chars if ch == '\n' or not ch.isspace() or ch == ' ']

    # 加入特殊 token
    special = ['<pad>', '<unk>']
    id2char = special + vocab_chars
    char2id = {ch: idx for idx, ch in enumerate(id2char)}

    vocab = {
        'char2id': char2id,
        'id2char': {str(i): ch for i, ch in enumerate(id2char)},
        'vocab_size': len(id2char),
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"词表大小: {len(id2char)}（含 <pad>, <unk>）")
    print(f"词表已写入: {vocab_path}")
    return char2id


# ──────────────────────────────────────────────
# 4. 数值化 → 保存 .npy
# ──────────────────────────────────────────────

def encode_corpus(corpus_path: str, char2id: dict, unk_id: int = 1) -> np.ndarray:
    """将 corpus.txt 转换为字符 ID 的 numpy 数组（int32）。"""
    print("数值化语料...")
    ids = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='编码'):
            for ch in line:
                ids.append(char2id.get(ch, unk_id))
    return np.array(ids, dtype=np.int32)


# ──────────────────────────────────────────────
# 5. 数据集划分
# ──────────────────────────────────────────────

def split_and_save(ids: np.ndarray, output_dir: str,
                   train_ratio: float = 0.8,
                   valid_ratio: float = 0.1):
    """
    按顺序切分为 train / valid / test，保存为 .npy 文件。
    默认 80% / 10% / 10%。
    """
    n = len(ids)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_ids = ids[:train_end]
    valid_ids = ids[train_end:valid_end]
    test_ids  = ids[valid_end:]

    output_dir = Path(output_dir)
    np.save(output_dir / 'train_ids.npy', train_ids)
    np.save(output_dir / 'valid_ids.npy', valid_ids)
    np.save(output_dir / 'test_ids.npy',  test_ids)

    print(f"划分完成：train={len(train_ids):,}  valid={len(valid_ids):,}  test={len(test_ids):,}")
    print(f"已保存至: {output_dir}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='中文新闻语料预处理')
    parser.add_argument('--input',      default='news_text',    help='输入文件夹路径')
    parser.add_argument('--output_dir', default='data/processed', help='输出文件夹路径')
    parser.add_argument('--limit',      type=int, default=None, help='只处理前 N 个文件（调试用）')
    parser.add_argument('--min_chinese',type=int, default=50,   help='文章最少汉字数，不足则过滤')
    parser.add_argument('--min_freq',   type=int, default=1,    help='词表最低字符频次')
    parser.add_argument('--seq_len',    type=int, default=128,  help='序列长度（仅供 dataset.py 使用，本脚本不切片）')
    parser.add_argument('--train_ratio',type=float,default=0.8, help='训练集比例')
    parser.add_argument('--valid_ratio',type=float,default=0.1, help='验证集比例')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / 'corpus.txt'
    vocab_path  = output_dir / 'vocab.json'

    # Step 1: 清洗 → corpus.txt
    build_corpus(args.input, str(corpus_path), limit=args.limit, min_chinese=args.min_chinese)

    # Step 2: 构建词表
    char2id = build_vocab(str(corpus_path), str(vocab_path), min_freq=args.min_freq)

    # Step 3: 数值化
    ids = encode_corpus(str(corpus_path), char2id)
    print(f"总 token 数: {len(ids):,}")

    # Step 4: 划分并保存
    split_and_save(ids, str(output_dir),
                   train_ratio=args.train_ratio,
                   valid_ratio=args.valid_ratio)

    print("\n全部完成！")
    print(f"  语料:     {corpus_path}")
    print(f"  词表:     {vocab_path}")
    print(f"  train:    {output_dir / 'train_ids.npy'}")
    print(f"  valid:    {output_dir / 'valid_ids.npy'}")
    print(f"  test:     {output_dir / 'test_ids.npy'}")
    print(f"\n后续使用 dataset.py 的 CharDataset 加载数据，seq_len={args.seq_len}")


if __name__ == '__main__':
    main()
