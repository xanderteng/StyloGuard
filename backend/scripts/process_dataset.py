from scripts.seed_db import seed_database


if __name__ == "__main__":
    count = seed_database()
    print(f"Processed raw CSV files and inserted {count} new article(s).")
