from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieve_augment_generate import RetAugGen

class Query(BaseModel):
    user_id: str = "shivaborusu"
    query: str
    k: int = 4

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/search")
def search(request: Query):
    try:
     response = RetAugGen().get_response(query=request.query)
     
     return {'user_id': request.user_id,
        "query": request.query,
        "response": response}
    except Exception as e:
       raise HTTPException(status_code=500,
                           detail="Failed to generate a response")