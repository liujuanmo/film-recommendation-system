import sys
from src.recommender import MovieRecommender

if __name__ == '__main__':
    # 示例：命令行参数可指定用户偏好
    # 例如: python main.py Action Drama 2010 director=Spielberg cast=Hanks keyword=war overview="epic battle" title="Saving"
    user_genres = []
    user_year = None
    user_directors = []
    user_cast = []
    user_keywords = []
    user_overview = None
    user_title = None
    for arg in sys.argv[1:]:
        if arg.startswith('director='):
            user_directors.append(arg.split('=', 1)[1])
        elif arg.startswith('cast='):
            user_cast.append(arg.split('=', 1)[1])
        elif arg.startswith('keyword='):
            user_keywords.append(arg.split('=', 1)[1])
        elif arg.startswith('overview='):
            user_overview = arg.split('=', 1)[1]
        elif arg.startswith('title='):
            user_title = arg.split('=', 1)[1]
        elif arg.isdigit():
            user_year = arg
        else:
            user_genres.append(arg)

    recommender = MovieRecommender()
    recommendations = recommender.recommend(
        genres=user_genres,
        year=user_year,
        directors=user_directors,
        cast=user_cast,
        keywords=user_keywords,
        overview=user_overview,
        title=user_title,
        top_n=10
    )
    print('Top Recommendations:')
    for idx, movie in enumerate(recommendations, 1):
        print(f"{idx}. {movie['title']} ({movie['year']}) | Genres: {movie['genres']} | Directors: {movie['directors']} | Cast: {movie['cast']}")
