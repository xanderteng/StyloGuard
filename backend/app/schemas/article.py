from pydantic import BaseModel, ConfigDict, field_validator

class ArticleBase(BaseModel):
    author: str
    title: str
    date: str
    category: str
    text: str
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if value.startswith(("http://", "https://", "missing:")):
            return value
        raise ValueError("url must be an HTTP(S) URL or a generated missing: sentinel")

class ArticleCreate(ArticleBase):
    pass

class Article(ArticleBase):
    id: int

    model_config = ConfigDict(from_attributes=True)
