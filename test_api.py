#!/usr/bin/env python3
"""
Test script for the Movie Recommendation API

This script demonstrates how to use the API endpoints.

Prerequisites:
1. Run 'python load_data.py' to load movie data
2. Run 'python load_embeddings.py' to compute embeddings  
3. Run 'python main.py' to start the FastAPI server
4. Run this test script: python test_api.py
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_get_recommendations():
    """Test GET recommendations with URL parameters."""
    print("🎬 Testing GET recommendations...")
    
    # Test with genres and year
    params = {
        "genres": "Action,Drama",
        "year": "2010",
        "top_n": 5
    }
    
    response = requests.get(f"{API_BASE_URL}/recommend", params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Query: {data['query_summary']}")
        print("Recommendations:")
        for i, movie in enumerate(data['recommendations'], 1):
            print(f"  {i}. {movie['title']} ({movie['year']}) - {movie['genres']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_post_recommendations():
    """Test POST recommendations with JSON body."""
    print("🎯 Testing POST recommendations...")
    
    payload = {
        "genres": ["Action", "Sci-Fi"],
        "year": "2010",
        "directors": ["Christopher Nolan"],
        "overview": "A mind-bending thriller about dreams and reality",
        "top_n": 3
    }
    
    response = requests.post(
        f"{API_BASE_URL}/recommend",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Query: {data['query_summary']}")
        print("Recommendations:")
        for i, movie in enumerate(data['recommendations'], 1):
            print(f"  {i}. {movie['title']} ({movie['year']})")
            print(f"     Directors: {movie['directors']}")
            print(f"     Cast: {movie['cast']}")
            print()
    else:
        print(f"Error: {response.text}")

def test_director_search():
    """Test director-specific search."""
    print("🎭 Testing director search...")
    
    params = {
        "directors": "Steven Spielberg",
        "top_n": 3
    }
    
    response = requests.get(f"{API_BASE_URL}/recommend", params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Query: {data['query_summary']}")
        print("Spielberg movies:")
        for i, movie in enumerate(data['recommendations'], 1):
            print(f"  {i}. {movie['title']} ({movie['year']})")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run all tests."""
    print("Movie Recommendation API Tests")
    print("=" * 50)
    print(f"Testing API at: {API_BASE_URL}")
    print("Make sure the FastAPI server is running: python main.py")
    print()
    
    try:
        test_health_check()
        test_get_recommendations()
        test_post_recommendations()
        test_director_search()
        
        print("✅ All tests completed!")
        print(f"Visit {API_BASE_URL} for the web interface")
        print(f"Visit {API_BASE_URL}/docs for interactive documentation")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed!")
        print("Make sure you've completed all setup steps:")
        print("  1. python load_data.py")
        print("  2. python load_embeddings.py")
        print("  3. python main.py")
        print("Then run this test script again.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 