import json
import random

INPUT_FILE = "../data/jsonl/vihallu-train.jsonl"
TRAIN_OUT = "../data/processed/vihallu-train-split.jsonl"
VAL_OUT = "../data/processed/vihallu-val-split.jsonl"

SPLIT_RATIO = 0.1
SEED = 42

def main():
    # Load toàn bộ dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Shuffle để random sample
    random.seed(SEED)
    random.shuffle(data)

    # Tính số mẫu validation
    n_total = len(data)
    n_val = int(n_total * SPLIT_RATIO)

    val_data = data[:n_val]
    train_data = data[n_val:]
 
    # Ghi ra file JSONL 
    with open(TRAIN_OUT, "w", encoding="utf-8") as f_train:
        for item in train_data:
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_OUT, "w", encoding="utf-8") as f_val:
        for item in val_data:
            f_val.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Tổng số mẫu: {n_total}")
    print(f"   Train: {len(train_data)} mẫu → {TRAIN_OUT}")
    print(f"   Val:   {len(val_data)} mẫu → {VAL_OUT}")

if __name__ == "__main__":
    main()