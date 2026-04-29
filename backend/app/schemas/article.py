from pydantic import BaseModel, HttpUrl
from typing import Optional

class ArticleBase(BaseModel):
    author: str
    title: str
    date: str
    category: str
    text: str
    url: Optional[HttpUrl] = None

class ArticleCreate(ArticleBase):
    pass

class Article(ArticleBase):
    id: int

    class Config:
        orm_mode = True
