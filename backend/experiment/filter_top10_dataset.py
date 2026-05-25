# -*- coding: utf-8 -*-
"""
Filters the merged dataset to include only the top 15 human authors and the AI author,
using the exact same preprocessing and ratio constraints as the StyloGuard benchmarks.
This is implemented in pure Python (standard library only) with CP1252-safe console printing.
"""

import csv
import re
import random
from collections import Counter
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_PATH = DATA_DIR / "merged_dataset.csv"
OUTPUT_PATH = DATA_DIR / "filtered_top10.csv"

# Configuration
SEED = 42
MAX_AUTHORS = 10
MAX_AI_RATIO_TO_HUMAN = 0.25

def clean_special_chars(text):
    text = str(text)
    text = re.sub(r'[\*\"”"“\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} does not exist.")
        return

    print(f"Loading merged dataset from: {INPUT_PATH}")
    
    # Read rows with utf-8-sig to automatically handle any UTF-8 BOM signatures
    rows = []
    with open(INPUT_PATH, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"Field names detected: {fieldnames}")
    print(f"Total original rows: {len(rows)}")

    # Clean text columns
    for row in rows:
        row["text"] = clean_special_chars(row["text"])

    # Separate human and AI
    human_rows = [r for r in rows if r["author"] != "AI"]
    ai_rows = [r for r in rows if r["author"] == "AI"]

    # Count human authors
    human_author_counts = Counter([r["author"] for r in human_rows])
    
    # Sort and pick top 10 human authors
    top_human_counts = human_author_counts.most_common(MAX_AUTHORS)
    top_human_authors = {author for author, _ in top_human_counts}

    print(f"\nSelected top {len(top_human_authors)} human authors:")
    for idx, (author, count) in enumerate(top_human_counts, 1):
        print(f"  {idx:02d}. {author} ({count} articles)")

    # Filter human articles to top 10
    human_filtered_rows = [r for r in human_rows if r["author"] in top_human_authors]

    # Calculate exact AI rows to sample (strictly 25% of human rows)
    max_ai_rows = int(len(human_filtered_rows) * MAX_AI_RATIO_TO_HUMAN)

    print(f"\nTarget AI sample size: {max_ai_rows} (Strictly 25% of human rows)")

    # Sample AI rows deterministically
    random.seed(SEED)
    sampled_ai_rows = random.sample(ai_rows, min(len(ai_rows), max_ai_rows))

    # Combine
    final_rows = human_filtered_rows + sampled_ai_rows

    # Shuffle
    random.shuffle(final_rows)

    # Save to file
    with open(OUTPUT_PATH, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n[SUCCESS] Filtered dataset successfully saved to: {OUTPUT_PATH}")
    print(f"Total Rows: {len(final_rows)} (Humans: {len(human_filtered_rows)}, AI: {len(sampled_ai_rows)})")

if __name__ == "__main__":
    main()
