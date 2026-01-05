import os
import sys
import json
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import our helpers
from src.utils.rag_helper import RAGHelper

load_dotenv()

app = FastAPI(title="Real-time Event Detection API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo, allow all. In prod, restrict to localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
engine = create_engine(DB_URL)
rag_helper = RAGHelper(db_url=DB_URL)

class Post(BaseModel):
    source: str
    content: str
    time: Optional[str] = None

class Trend(BaseModel):
    id: int
    trend_name: str
    trend_score: float
    category: Optional[str]
    summary: Optional[str]
    last_updated: datetime
    post_count: int
    representative_posts: List[Post]

@app.get("/")
def read_root():
    return {"status": "online", "message": "Cyber Intelligence API v1"}

@app.get("/trends", response_model=List[Trend])
def get_trends(limit: int = 50, min_score: float = 0.0):
    query = text("""
        SELECT id, trend_name, trend_score, category, summary, last_updated, post_count, representative_posts 
        FROM detected_trends 
        WHERE trend_score >= :min_score
        ORDER BY last_updated DESC 
        LIMIT :limit
    """)
    
    with engine.connect() as conn:
        results = conn.execute(query, {"min_score": min_score, "limit": limit}).fetchall()
        
    trends = []
    for r in results:
        reps_raw = r.representative_posts
        reps = json.loads(reps_raw) if isinstance(reps_raw, str) else (reps_raw or [])
        
        trends.append(Trend(
            id=r.id,
            trend_name=r.trend_name,
            trend_score=r.trend_score,
            category=r.category,
            summary=r.summary,
            last_updated=r.last_updated,
            post_count=r.post_count,
            representative_posts=[Post(**p) for p in reps[:5]]
        ))
    return trends

@app.get("/trends/{trend_id}", response_model=Trend)
def get_trend_detail(trend_id: int):
    query = text("SELECT * FROM detected_trends WHERE id = :id")
    with engine.connect() as conn:
        r = conn.execute(query, {"id": trend_id}).fetchone()
    
    if not r:
        raise HTTPException(status_code=404, detail="Trend not found")
        
    reps_raw = r.representative_posts
    reps = json.loads(reps_raw) if isinstance(reps_raw, str) else (reps_raw or [])
    
    return Trend(
        id=r.id,
        trend_name=r.trend_name,
        trend_score=r.trend_score,
        category=r.category,
        summary=r.summary,
        last_updated=r.last_updated,
        post_count=r.post_count,
        representative_posts=[Post(**p) for p in reps]
    )

@app.get("/analytics/semantic-map")
def get_semantic_map():
    df = rag_helper.get_semantic_map_data(limit=150)
    if df is None:
        return {"data": []}
    return {"data": df.to_dict(orient="records")}

@app.post("/chat")
def chat_with_ai(query: str):
    relevant_trends = rag_helper.get_relevant_trends(query, top_k=3)
    if not relevant_trends:
        return {"answer": "Không tìm thấy thông tin liên quan."}
    
    answer = rag_helper.generate_answer(query, relevant_trends)
    return {
        "answer": answer,
        "sources": [t['name'] for t in relevant_trends]
    }

@app.get("/report/generate")
def generate_report():
    report = rag_helper.generate_daily_report()
    return {"report": report}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
