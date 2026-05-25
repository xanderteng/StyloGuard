from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Article
from app.dependencies import get_db
from app.schemas.article import Article as ArticleSchema

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("", response_model=list[ArticleSchema])
def list_articles(
    db: Session = Depends(get_db),
    author: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[Article]:
    query = db.query(Article)
    if author:
        query = query.filter(Article.author.ilike(author.strip()))
    return query.order_by(Article.id).offset(offset).limit(limit).all()


@router.get("/authors")
def list_authors(db: Session = Depends(get_db), min_articles: int = Query(1, ge=1)) -> list[dict]:
    rows = (
        db.query(Article.author, func.count(Article.id).label("article_count"))
        .group_by(Article.author)
        .having(func.count(Article.id) >= min_articles)
        .order_by(func.count(Article.id).desc(), Article.author.asc())
        .all()
    )
    return [{"author": author, "article_count": count} for author, count in rows]


@router.get("/count")
def count_articles(db: Session = Depends(get_db), author: str | None = None) -> dict:
    query = db.query(Article)
    if author:
        query = query.filter(Article.author.ilike(author.strip()))
    return {"count": query.count()}


@router.get("/{article_id}", response_model=ArticleSchema)
def get_article(article_id: int, db: Session = Depends(get_db)) -> Article:
    article = db.query(Article).filter(Article.id == article_id).first()
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article
