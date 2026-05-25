import csv
import hashlib
import logging
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from app.db.models import Article
from app.db.session import Base, SessionLocal, engine

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
BATCH_SIZE = 100
logger = logging.getLogger(__name__)


def _commit_batch(db, records: list[dict], existing_urls: set[str]) -> int:
    if not records:
        return 0

    try:
        db.commit()
        existing_urls.update(record["url"] for record in records)
        return len(records)
    except IntegrityError as exc:
        db.rollback()
        logger.warning("Batch commit failed for %s record(s); retrying individually: %s", len(records), exc)

    inserted = 0
    for record in records:
        if record["url"] in existing_urls:
            continue
        db.add(Article(**record))
        try:
            db.commit()
            existing_urls.add(record["url"])
            inserted += 1
        except IntegrityError as exc:
            db.rollback()
            logger.warning("Skipping duplicate or invalid article URL %s: %s", record["url"], exc)

    return inserted


def seed_database() -> int:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    inserted = 0
    pending_records: list[dict] = []
    pending_urls: set[str] = set()

    try:
        existing_urls = {url for (url,) in db.query(Article.url).all()}
        target_csv_path = PROCESSED_DATA_DIR / "filtered_top10.csv"
        csv_paths = [target_csv_path] if target_csv_path.exists() else sorted(PROCESSED_DATA_DIR.glob("*.csv"))
        
        for csv_path in csv_paths:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    url = (row.get("url") or "").strip()
                    if not url:
                        fallback_source = "|".join(
                            [
                                csv_path.name,
                                row.get("author") or "",
                                row.get("title") or "",
                                row.get("text") or "",
                            ]
                        )
                        fallback_hash = hashlib.sha256(fallback_source.encode("utf-8")).hexdigest()[:16]
                        url = f"missing:{csv_path.name}:{fallback_hash}"
                    if url in existing_urls or url in pending_urls:
                        continue

                    record = {
                        "author": (row.get("author") or "Unknown").strip() or "Unknown",
                        "title": (row.get("title") or "Untitled").strip() or "Untitled",
                        "date": (row.get("date") or "").strip(),
                        "category": (row.get("category") or "uncategorized").strip() or "uncategorized",
                        "text": (row.get("text") or "").strip(),
                        "url": url,
                    }
                    if not record["text"]:
                        continue
                    db.add(Article(**record))
                    pending_records.append(record)
                    pending_urls.add(url)

                    if len(pending_records) >= BATCH_SIZE:
                        inserted += _commit_batch(db, pending_records, existing_urls)
                        pending_records.clear()
                        pending_urls.clear()

        if pending_records:
            inserted += _commit_batch(db, pending_records, existing_urls)
        return inserted
    finally:
        db.close()


if __name__ == "__main__":
    count = seed_database()
    print(f"Inserted {count} article(s).")
