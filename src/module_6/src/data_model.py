from pydantic import BaseModel


class Request(BaseModel):
    user_id: str


class Response(BaseModel):
    basket_price: float


class HTTPError(BaseModel):
    detail: str
