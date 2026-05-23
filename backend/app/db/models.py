from sqlalchemy import Column, Integer, String, Text, UniqueConstraint
from app.db.session import Base

class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (UniqueConstraint("url", name="uq_articles_url"),)

    id = Column(Integer, primary_key=True, index=True)
    author = Column(String, index=True)
    title = Column(String, index=True)
    date = Column(String)
    category = Column(String, index=True)
    text = Column(Text)
    url = Column(String, nullable=False)
