# Chinese Character-Level Language Model

A comparison of RNN, LSTM, and Transformer architectures for Chinese text generation, trained on Chinese news articles. Type a few characters and the model will continue writing automatically.

Pre-trained models are included — download and run immediately.

---

## Project Structure

```
├── preprocess.py          # Data preprocessing (cleaning, vocab building, encoding)
├── dataset.py             # PyTorch Dataset definition
├── model.py               # RNN / LSTM / Transformer model definitions
├── train.py               # Unified training script with early stopping
├── generate.py            # Text generation script
├── checkpoints/
│   ├── rnn.pt             # Trained RNN model
│   ├── lstm.pt            # Trained LSTM model
│   └── transformer.pt     # Trained Transformer model
└── data/processed/
    ├── corpus.txt         # Cleaned plain-text corpus
    ├── vocab.json         # Character-level vocabulary
    ├── train_ids.npy      # Training set
    ├── valid_ids.npy      # Validation set
    └── test_ids.npy       # Test set
```

---

## Requirements

- Python 3.10
- torch 2.5.1+cu124 (requires CUDA 12.4+, see below for CPU-only setup)
- numpy 1.26.4
- tqdm 4.67.3

**Install with GPU (CUDA 12.4+):**

```bash
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4 tqdm==4.67.3
```

**Install without GPU (CPU only):**

```bash
pip install torch==2.5.1 numpy==1.26.4 tqdm==4.67.3
```

> Generation runs at normal speed on CPU; training is significantly slower.

---

## Text Generation (Ready to Use)

```bash
# Generate with RNN
python generate.py --model rnn --prompt "今天"

# Generate with LSTM
python generate.py --model lstm --prompt "北京时间"

# Generate with Transformer
python generate.py --model transformer --prompt "中国队"
```

**Parameters:**

| Argument | Description | Default |
|---|---|---|
| `--model` | Model type: `rnn` / `lstm` / `transformer` | `rnn` |
| `--prompt` | Input prompt text | `今天` |
| `--length` | Number of characters to generate | `200` |
| `--temperature` | Sampling temperature — lower is more conservative, higher is more random | `1.0` |

**Examples:**

```bash
# Generate 300 characters with temperature 0.8 (smoother output)
python generate.py --model lstm --prompt "北京时间" --length 300 --temperature 0.8

# Greedy decoding (most conservative)
python generate.py --model transformer --prompt "中国队" --temperature 0.0

# Compare all three models on the same prompt
for model in rnn lstm transformer; do
    echo "===== $model ====="
    python generate.py --model $model --prompt "今天的比赛" --length 150 --temperature 0.8
done
```

---

## Train from Scratch (Optional)

If you want to retrain on your own data, follow these steps.

**Step 1: Prepare data**

Place your `.txt` news articles (one file per article) in the `news_text/` directory, then run preprocessing:

```bash
python preprocess.py --input news_text --output_dir data/processed
```

**Step 2: Train the models**

```bash
python train.py --model rnn
python train.py --model lstm
python train.py --model transformer
```

**Training parameters:**

| Argument | Description | Default |
|---|---|---|
| `--model` | Model type | `rnn` |
| `--seq_len` | Sequence length | `128` |
| `--batch_size` | Batch size | `64` |
| `--lr` | Learning rate | `1e-3` |
| `--patience` | Early stopping patience | `5` |

Training uses early stopping: if validation loss does not improve for `patience` consecutive epochs, training stops and the best checkpoint is saved to `checkpoints/`.
