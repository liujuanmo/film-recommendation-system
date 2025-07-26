from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from src.recommender import MovieRecommender
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="A movie recommendation system using vector similarity search with PostgreSQL and pgvector",
    version="1.0.0"
)

# Initialize the recommender (this will load/compute embeddings if needed)
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
        logger.error("2. Run 'python load_embeddings.py' to compute and store embeddings")
        raise e

# Pydantic models for request/response
class MovieRecommendation(BaseModel):
    """Response model for a single movie recommendation."""
    title: str = Field(description="Movie title")
    year: str = Field(description="Release year")
    genres: str = Field(description="Movie genres")
    directors: str = Field(description="Movie directors")
    cast: str = Field(description="Main cast")

class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[MovieRecommendation]
    query_summary: str = Field(description="Summary of the search query")

@app.get("/")
async def root():
    """Root endpoint - redirects to Swagger documentation."""
    return {"message": "Movie Recommendation API", "documentation": "/docs", "redoc": "/redoc"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Movie Recommendation API is running"}

@app.get("/recommend", response_model=RecommendationResponse)
async def recommend_movies(
    genres: Optional[str] = Query(None, description="Comma-separated list of genres", example="Action,Drama"),
    year: Optional[str] = Query(None, description="Preferred year", example="2010"),
    directors: Optional[str] = Query(None, description="Comma-separated list of directors", example="Christopher Nolan,Steven Spielberg"),
    cast: Optional[str] = Query(None, description="Comma-separated list of cast members", example="Leonardo DiCaprio,Tom Hanks"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords", example="heist,dreams"),
    overview: Optional[str] = Query(None, description="Plot description", example="A team of thieves who steal corporate secrets"),
    title: Optional[str] = Query(None, description="Title keywords", example="Inception"),
    top_n: Optional[int] = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """Get movie recommendations using query parameters."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        # Convert comma-separated strings to lists
        genres_list = genres.split(',') if genres else None
        directors_list = directors.split(',') if directors else None
        cast_list = cast.split(',') if cast else None
        keywords_list = keywords.split(',') if keywords else None
        
        # Get recommendations
        recommendations = recommender.recommend(
            genres=genres_list,
            year=year,
            directors=directors_list,
            cast=cast_list,
            keywords=keywords_list,
            overview=overview,
            title=title,
            top_n=top_n
        )
        
        # Create query summary
        query_parts = []
        if genres_list:
            query_parts.append(f"genres: {', '.join(genres_list)}")
        if year:
            query_parts.append(f"year: {year}")
        if directors_list:
            query_parts.append(f"directors: {', '.join(directors_list)}")
        if cast_list:
            query_parts.append(f"cast: {', '.join(cast_list)}")
        if overview:
            query_parts.append(f"plot: {overview}")
        if title:
            query_parts.append(f"title: {title}")
        if keywords_list:
            query_parts.append(f"keywords: {', '.join(keywords_list)}")
        
        query_summary = "; ".join(query_parts) if query_parts else "all movies"
        
        movie_recommendations = [
            MovieRecommendation(**movie) for movie in recommendations
        ]
        
        return RecommendationResponse(
            recommendations=movie_recommendations,
            query_summary=query_summary
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
