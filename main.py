from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List, Dict
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
        logger.error("1. Run 'python extract.py' to create merged.csv")
        logger.error("2. Run 'python embedding.py' to generate embeddings")
        logger.error("3. Run 'python save_to_db.py' to save data to database")
        raise e


@app.get("/")
async def root():
    """Root endpoint - redirects to Swagger documentation."""
    return {
        "message": "Movie Recommendation API",
        "documentation": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "text_recommendations": "/recommend/text",
            "filter_recommendations": "/recommend/filter",
            "similar_movies": "/recommend/similar/{movie_id}",
            "search_movies": "/search",
            "movie_details": "/movie/{movie_id}",
            "popular_genres": "/genres/popular",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Movie Recommendation API is running"}


@app.get("/recommend/text")
async def recommend_by_text(
    query: str = Query(
        ...,
        description="Natural language query",
        example="action movies with explosions",
    ),
    limit: int = Query(10, description="Number of recommendations", ge=1, le=50),
):
    """Get movie recommendations based on natural language query."""
    try:
        recommendations = recommender.recommend_by_text(query, limit=limit)

        return {
            "query": query,
            "recommendations": recommendations,
            "total_results": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/filter")
async def recommend_by_filters(
    genres: Optional[List[str]] = Query(
        None, description="List of genres", example=["Action", "Adventure"]
    ),
    year: Optional[int] = Query(None, description="Year filter", example=2020),
    directors: Optional[List[str]] = Query(
        None, description="List of directors", example=["Christopher Nolan"]
    ),
    actors: Optional[List[str]] = Query(
        None, description="List of actors", example=["Tom Hanks"]
    ),
    limit: int = Query(10, description="Number of recommendations", ge=1, le=50),
):
    """Get movie recommendations based on specific filters."""
    try:
        # Build filter parameters
        filters = {}
        if genres:
            filters["genres"] = genres
        if year:
            filters["year"] = year
        if directors:
            filters["directors"] = directors
        if actors:
            filters["actors"] = actors

        recommendations = recommender.recommend_by_filters(**filters, limit=limit)

        return {
            "filters": filters,
            "recommendations": recommendations,
            "total_results": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/similar/{movie_id}")
async def recommend_similar_movies(
    movie_id: str = Path(..., description="Movie ID (tconst)", example="tt0111161"),
    limit: int = Query(10, description="Number of recommendations", ge=1, le=50),
):
    """Get movies similar to a specific movie."""
    try:
        recommendations = recommender.recommend_similar_to_movie(movie_id, limit=limit)

        return {
            "movie_id": movie_id,
            "recommendations": recommendations,
            "total_results": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_movies(
    query: str = Query(..., description="Search term", example="Batman"),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
):
    """Search movies by title."""
    try:
        results = recommender.search_movies(query, limit=limit)

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movie/{movie_id}")
async def get_movie_details(
    movie_id: str = Path(..., description="Movie ID (tconst)", example="tt0111161"),
):
    """Get detailed information about a specific movie."""
    try:
        movie = recommender.get_movie_details(movie_id)

        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")

        return movie
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/genres/popular")
async def get_popular_genres(
    limit: int = Query(10, description="Number of genres", ge=1, le=50),
):
    """Get most popular genres in the database."""
    try:
        genres = recommender.get_popular_genres(limit=limit)

        return {
            "genres": genres,
            "total_genres": len(genres),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        # Get popular genres for stats
        genres = recommender.get_popular_genres(limit=5)

        return {
            "system": "Movie Recommendation API",
            "version": "1.0.0",
            "status": "healthy",
            "top_genres": genres,
            "features": [
                "Text-based recommendations",
                "Filter-based recommendations",
                "Similar movie recommendations",
                "Movie search",
                "Popular genres analysis",
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
