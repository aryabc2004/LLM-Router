from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
import router as router_module

app = FastAPI(title="LLM Cost Router")


class QueryRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def startup():
    db.get_conn()  # ensures schema exists
    try:
        router_module.get_classifier()
    except FileNotFoundError as e:
        # Don't crash the server; surface a clear error on first query instead.
        print(f"WARNING: {e}")


@app.post("/query")
def query(req: QueryRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    try:
        return router_module.handle_query(req.prompt)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    return db.get_stats()


@app.get("/health")
def health():
    return {"status": "ok"}
