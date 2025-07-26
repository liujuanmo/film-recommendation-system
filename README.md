
# Movie Recommendation System

A modern movie recommendation system using vector similarity search with PostgreSQL and pgvector. This system builds embeddings from movie features and provides a FastAPI web interface for getting personalized movie recommendations.

## 🎯 Features

- **Genre-based recommendations**: Find movies by preferred genres
- **Year filtering**: Get recommendations from specific time periods  
- **Director preferences**: Discover movies by favorite directors
- **Cast-based search**: Find movies featuring specific actors
- **Keyword matching**: Search using plot keywords
- **Overview similarity**: Match movies by plot descriptions
- **Title matching**: Find similar movies by title
- **FastAPI Web Interface**: Modern REST API with interactive documentation
- **PostgreSQL + pgvector**: High-performance vector similarity search

## 🚀 Quick Start

### Prerequisites

- PostgreSQL 12+ with pgvector extension installed
- Python 3.7+
- IMDB dataset files

**Note**: The system uses a custom SQLAlchemy Vector type, so you only need the PostgreSQL pgvector extension, not the Python pgvector package.

### Installation

1. **Set up PostgreSQL with pgvector** (see [SETUP_POSTGRESQL.md](SETUP_POSTGRESQL.md) for detailed instructions)

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download IMDB data** from [IMDB datasets](https://datasets.imdbws.com/) and place TSV files in `imdb_data/` directory:
   - `title.basics.tsv`
   - `title.crew.tsv`
   - `title.principals.tsv`
   - `name.basics.tsv`

### Usage

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Load movie data into PostgreSQL**
```bash
python load_data.py
```

**Step 3: Compute and store embeddings**
```bash
python load_embeddings.py
```

**Step 4: Start the FastAPI server**
```bash
python main.py
```

**Step 5: Use the API**
- Interactive Swagger docs: http://localhost:8000/docs
- Alternative ReDoc documentation: http://localhost:8000/redoc
- API root: http://localhost:8000

## 📡 API Usage Examples

### GET Requests (URL Parameters)
```bash
# Action and Drama movies from 2010
curl "http://localhost:8000/recommend?genres=Action,Drama&year=2010"

# Movies by Christopher Nolan
curl "http://localhost:8000/recommend?directors=Christopher%20Nolan"

# Space exploration movies
curl "http://localhost:8000/recommend?overview=space%20exploration"

# Combined search
curl "http://localhost:8000/recommend?genres=Action,Thriller&year=2010&directors=Christopher%20Nolan"
```

### POST Requests (JSON Body)
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "genres": ["Action", "Sci-Fi"],
       "year": "2010", 
       "directors": ["Christopher Nolan"],
       "overview": "A mind-bending thriller about dreams",
       "top_n": 5
     }'
```

### Response Format
```json
{
  "recommendations": [
    {
      "title": "Inception",
      "year": "2010",
      "genres": "Action,Sci-Fi,Thriller", 
      "directors": "Christopher Nolan",
      "cast": "Leonardo DiCaprio,Marion Cotillard,Ellen Page"
    }
  ],
  "query_summary": "genres: Action, Sci-Fi; year: 2010; directors: Christopher Nolan"
}
```

## 🏗️ Architecture

The system uses a clean three-step architecture:

### 1. Data Layer (`load_data.py`)
- Loads IMDB datasets into normalized PostgreSQL tables
- Creates proper relationships between movies, directors, and actors
- Sets up database indexes for performance

### 2. Embedding Layer (`load_embeddings.py`)
- Computes vector embeddings from movie features
- Stores embeddings in PostgreSQL with pgvector indexing
- Handles genre, director, cast, and text feature engineering

### 3. API Layer (`main.py`)
- FastAPI web server with automatic documentation
- Real-time similarity search using precomputed embeddings
- Query processing and recommendation generation

### Core Components
- `main.py`: FastAPI application with recommendation endpoints
- `load_data.py`: Data loading script for PostgreSQL
- `load_embeddings.py`: Embedding computation and storage
- `src/recommender.py`: Core recommendation logic
- `src/postgresql_vec_client.py`: SQLAlchemy models and database operations
- `src/feature_engineering.py`: Text feature extraction
- `src/embedding_utils.py`: Embedding utilities

## 🔧 Technology Stack

- **Database**: PostgreSQL with pgvector extension
- **ORM**: SQLAlchemy for database operations
- **Backend**: FastAPI with Pydantic models
- **ML/AI**: scikit-learn, numpy, pandas
- **Vector Search**: pgvector (PostgreSQL extension) with custom SQLAlchemy types
- **API Documentation**: Automatic Swagger/OpenAPI docs

## 📊 How It Works

1. **Data Ingestion**: IMDB datasets → PostgreSQL tables (`load_data.py`)
2. **Feature Engineering**: Movies → Genre/Director/Cast/Text embeddings (`load_embeddings.py`)
3. **Vector Storage**: Embeddings → pgvector-indexed table (`load_embeddings.py`)
4. **API Serving**: FastAPI server initialization (`main.py`)
5. **Query Processing**: User preferences → Query vector (`main.py`)
6. **Similarity Search**: Cosine similarity → Top-N recommendations (`main.py`)

## 🔗 Benefits Over SQLite

- **Performance**: Better handling of large datasets with PostgreSQL
- **Concurrency**: Multiple users can query simultaneously
- **Scalability**: Production-ready PostgreSQL infrastructure with SQLAlchemy ORM
- **Flexibility**: Type-safe database operations with SQLAlchemy models
- **Maintainability**: Clean ORM-based database interactions
- **Reliability**: ACID compliance and backup capabilities

## 🚀 Development Setup

For development with auto-reload:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload
```

## 🔧 Troubleshooting

### SQLAlchemy Import Error
If you see `ImportError: cannot import name 'VECTOR'`, this is expected. The system uses a custom Vector type that works with the PostgreSQL pgvector extension without requiring additional Python packages.

### Vector Search SQL Syntax
The system uses raw psycopg2 connections for vector similarity queries to avoid SQLAlchemy parameter binding issues with PostgreSQL's `::vector` casting syntax.

### Missing Dependencies
Make sure to install all requirements:
```bash
pip install -r requirements.txt
```

### PostgreSQL Connection Issues
1. Ensure PostgreSQL is running
2. Verify the pgvector extension is installed: `CREATE EXTENSION vector;`
3. Check database connection settings via environment variables

### Vector Search Errors
If you see SQL syntax errors related to vector operations, ensure:
1. The pgvector extension is properly installed in PostgreSQL
2. Your PostgreSQL version supports pgvector (12+)
3. The vector data format is correct (arrays converted to strings)