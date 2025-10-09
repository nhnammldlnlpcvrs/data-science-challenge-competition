import csv
import json
import os

CSV_DIR = "../data/csv"
JSONL_DIR = "../data/jsonl"

os.makedirs(JSONL_DIR, exist_ok=True)

def convert_csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, "r", encoding="utf-8") as f_csv, \
         open(jsonl_file, "w", encoding="utf-8") as f_jsonl:
        
        reader = csv.DictReader(f_csv)
        for row in reader:
            f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Đã chuyển {csv_file} → {jsonl_file}")

def main():
    files = ["vihallu-train.csv", "vihallu-public-test.csv", "vihallu-private-test.csv"]
    for file in files:
        csv_path = os.path.join(CSV_DIR, file)
        jsonl_path = os.path.join(JSONL_DIR, file.replace(".csv", ".jsonl"))
        convert_csv_to_jsonl(csv_path, jsonl_path)

if __name__ == "__main__":
    main()
 