from sqlalchemy import Column, Integer, String, Text
from app.db.session import Base

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    author = Column(String, index=True)
    title = Column(String, index=True)
    date = Column(String)
    category = Column(String, index=True)
    text = Column(Text)
    url = Column(String)
