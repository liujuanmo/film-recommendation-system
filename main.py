from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List
import logging

from src.recommender import MovieRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="A semantic movie recommendation system using PostgreSQL and sentence transformers",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender = None


@app.on_event("startup")
async def startup_event():
    """Initialize the movie recommender on startup."""
    global recommender
    try:
        logger.info("Initializing movie recommender...")
        recommender = MovieRecommender()
        logger.info("Movie recommender initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        logger.error("Make sure you've completed the setup steps:")
        logger.error("1. Run 'python load_data.py' to load movie data into PostgreSQL")
        logger.error(
            "2. Run 'python load_embeddings.py' to compute and store embeddings"
        )
        raise e


@app.get("/")
async def root():
    """Root endpoint - redirects to Swagger documentation."""
    return {
        "message": "Movie Recommendation API",
        "documentation": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Movie Recommendation API is running"}


@app.get("/recommend")
async def recommend_movies(
    query: str = Query(
        ..., description="Search query", example="I like action movies from 2010"
    ),
    year: Optional[int] = Query(None, description="Year filter", example=2010),
    genre: Optional[str] = Query(None, description="Genre filter", example="Action"),
    limit: int = Query(10, description="Number of recommendations", ge=1, le=50),
):
    """Get movie recommendations based on natural language query."""
    try:
        # Build query parts
        query_parts = [query]

        if year:
            query_parts.append(f"year: {year}")

        if genre:
            query_parts.append(f"genre: {genre}")

        # Combine all query parts
        full_query = " ".join(query_parts)

        # Get recommendations
        recommendations = recommender.recommend(full_query, limit=limit)

        return {
            "query": query,
            "recommendations": recommendations,
            "total_results": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
