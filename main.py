from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
import router as router_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before `yield` runs once when the server starts;
    # code after it would run once when the server shuts down.
    db.get_conn().close()  # ensures schema exists
    try:
        router_module.get_classifier()
    except FileNotFoundError as e:
        # Don't crash the server; surface a clear error on first query instead.
        print(f"WARNING: {e}")
    yield


app = FastAPI(title="LLM Cost Router", lifespan=lifespan)


class QueryRequest(BaseModel):
    prompt: str


@app.post("/query")
def query(req: QueryRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    try:
        return router_module.handle_query(req.prompt)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats(real_only: bool = False):
    """Pass ?real_only=true to exclude mock-mode rows from the totals."""
    return db.get_stats(real_only=real_only)


@app.get("/health")
def health():
    return {"status": "ok"}
