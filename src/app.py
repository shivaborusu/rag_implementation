from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from query_response import get_query_response

class Query(BaseModel):
    user_id: str
    query: str
    k: int = 4

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/search")
def search(request: Query):
    response = get_query_response(request.query)
    return {'user_id': request.user_id,
            "query": request.query,
            "respinse": response}

