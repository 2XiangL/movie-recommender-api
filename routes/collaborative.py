from flask import Blueprint, request, jsonify
import sys
import os
from pathlib import Path
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.collaborative_filtering.cf_data_processor import CollaborativeDataProcessor
from utils.collaborative_filtering.cf_recommender import CollaborativeFilteringRecommender
from utils.common import convert_numpy_types
from config.settings import Config

collaborative_bp = Blueprint('collaborative', __name__)

# 全局推荐器实例
cf_recommender = None
data_processor = None
movies_df = None
ratings_df = None

def get_cf_system():
    """获取协同过滤系统实例（延迟加载）"""
    global cf_recommender, data_processor, movies_df, ratings_df
    if cf_recommender is None:
        try:
            config = Config.COLLABORATIVE_CONFIG
            data_processor = CollaborativeDataProcessor(config)
            movies_df, ratings_df, _ = data_processor.load_processed_data()

            # 加载训练好的模型
            model_path = os.path.join(config["data"]["data_directory"], config["files"]["cf_model"])
            cf_recommender = CollaborativeFilteringRecommender()
            cf_recommender.load_model(model_path)

        except Exception as e:
            raise Exception(f"Failed to initialize collaborative filtering system: {e}")
    return cf_recommender, movies_df, ratings_df

@collaborative_bp.route('/recommend-user/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """
    为用户推荐电影

    Path Parameters:
    - user_id: 用户ID

    Query Parameters:
    - n: 推荐数量 (可选，默认为10)
    """
    try:
        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        rec, movies_df, ratings_df = get_cf_system()
        recommendations = rec.recommend_for_user(
            user_id=user_id,
            n_recommendations=n,
            exclude_seen=True,
            ratings_df=ratings_df
        )

        if not recommendations:
            return jsonify({
                "error": f"Unable to generate recommendations for user {user_id}",
                "suggestions": "User ID may not exist or user has rated most movies"
            }), 404

        # Convert numpy types to Python native types
        recommendations = convert_numpy_types(recommendations)

        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@collaborative_bp.route('/similar-movies/<int:movie_id>', methods=['GET'])
def similar_movies(movie_id):
    """
    查找相似电影

    Path Parameters:
    - movie_id: 电影ID

    Query Parameters:
    - n: 结果数量 (可选，默认为10)
    """
    try:
        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        rec, movies_df, ratings_df = get_cf_system()

        # 检查电影是否存在
        movie_info = movies_df[movies_df['movie_id'] == movie_id]
        if movie_info.empty:
            return jsonify({"error": f"Movie with ID {movie_id} not found"}), 404

        similar = rec.find_similar_movies(movie_id, n)

        if not similar:
            return jsonify({"error": f"No similar movies found for movie ID {movie_id}"}), 404

        # Convert numpy types to Python native types
        similar = convert_numpy_types(similar)

        return jsonify({
            "movie_id": convert_numpy_types(movie_id),
            "movie_title": movie_info.iloc[0]['title'],
            "similar_movies": similar,
            "count": len(similar)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@collaborative_bp.route('/find-similar-by-name', methods=['GET'])
def find_similar_by_name():
    """
    通过电影名称查找相似电影

    Query Parameters:
    - movie_name: 电影名称 (必需)
    - n: 结果数量 (可选，默认为10)
    - fuzzy: 是否使用模糊匹配 (可选，默认为True)
    """
    def convert_numpy_types(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    try:
        movie_name = request.args.get('movie_name')
        if not movie_name:
            return jsonify({"error": "Missing required parameter: movie_name"}), 400

        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        fuzzy = request.args.get('fuzzy', type=lambda x: x.lower() == 'true', default=True)

        rec, movies_df, ratings_df = get_cf_system()

        # 搜索电影
        if fuzzy:
            # 模糊匹配
            movie_matches = movies_df[
                movies_df['title'].str.contains(movie_name, case=False, na=False)
            ]
        else:
            # 精确匹配
            movie_matches = movies_df[
                movies_df['title'].str.lower() == movie_name.lower()
            ]

        if movie_matches.empty:
            return jsonify({
                "error": f"Movie '{movie_name}' not found",
                "suggestions": "Try using fuzzy search or check the movie name spelling"
            }), 404

        # 如果找到多个匹配，使用第一个
        target_movie = movie_matches.iloc[0]
        movie_id = target_movie['movie_id']
        movie_title = target_movie['title']

        # 查找相似电影
        similar = rec.find_similar_movies(movie_id, n)

        if not similar:
            return jsonify({
                "error": f"No similar movies found for '{movie_title}'",
                "movie_id": convert_numpy_types(movie_id),
                "movie_title": movie_title
            }), 404

        # Convert numpy types to Python native types
        similar = convert_numpy_types(similar)

        return jsonify({
            "movie_id": convert_numpy_types(movie_id),
            "movie_title": movie_title,
            "similar_movies": similar,
            "count": len(similar),
            "search_info": {
                "query": movie_name,
                "fuzzy_search": fuzzy,
                "total_matches": len(movie_matches)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@collaborative_bp.route('/similar-users/<int:user_id>', methods=['GET'])
def similar_users(user_id):
    """
    查找相似用户

    Path Parameters:
    - user_id: 用户ID

    Query Parameters:
    - n: 结果数量 (可选，默认为10)
    """
    try:
        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        rec, movies_df, ratings_df = get_cf_system()
        similar = rec.find_similar_users(user_id, n)

        if not similar:
            return jsonify({"error": f"No similar users found for user ID {user_id}"}), 404

        # Convert numpy types to Python native types
        similar = convert_numpy_types(similar)

        return jsonify({
            "user_id": user_id,
            "similar_users": similar,
            "count": len(similar)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@collaborative_bp.route('/user-profile/<int:user_id>', methods=['GET'])
def user_profile(user_id):
    """
    获取用户画像

    Path Parameters:
    - user_id: 用户ID
    """
    try:
        rec, movies_df, ratings_df = get_cf_system()
        profile = rec.get_user_profile(user_id, ratings_df)

        if 'error' in profile:
            return jsonify({"error": profile['error']}), 404

        return jsonify({
            "user_id": user_id,
            "profile": profile
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@collaborative_bp.route('/search-movies', methods=['GET'])
def search_movies():
    """
    搜索电影

    Query Parameters:
    - q: 搜索关键词 (必需)
    - n: 结果数量 (可选，默认为10)
    """
    try:
        query = request.args.get('q')
        if not query:
            return jsonify({"error": "Missing required parameter: q"}), 400

        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        _, movies_df, _ = get_cf_system()

        # 在电影标题中搜索
        matches = movies_df[
            movies_df['title'].str.contains(query, case=False, na=False)
        ]

        if matches.empty:
            return jsonify({
                "query": query,
                "results": [],
                "count": 0
            })

        # 转换为字典格式
        results = []
        for _, movie in matches.head(n).iterrows():
            results.append({
                "movie_id": convert_numpy_types(movie['movie_id']),
                "title": movie['title'],
                "vote_average": convert_numpy_types(movie['vote_average']),
                "vote_count": convert_numpy_types(movie['vote_count'])
            })

        # Convert numpy types to Python native types
        results = convert_numpy_types(results)

        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@collaborative_bp.route('/top-users', methods=['GET'])
def top_users():
    """
    获取最活跃用户

    Query Parameters:
    - n: 用户数量 (可选，默认为10)
    """
    try:
        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 100:
            return jsonify({"error": "Parameter n must be between 1 and 100"}), 400

        _, movies_df, ratings_df = get_cf_system()

        user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        user_stats.columns = ['rating_count', 'avg_rating', 'rating_std']
        user_stats = user_stats.sort_values('rating_count', ascending=False).head(n)

        # 转换为字典格式
        users = []
        for user_id, stats in user_stats.iterrows():
            users.append({
                "user_id": convert_numpy_types(user_id),
                "rating_count": convert_numpy_types(stats['rating_count']),
                "avg_rating": convert_numpy_types(stats['avg_rating']),
                "rating_std": convert_numpy_types(stats['rating_std']) if not pd.isna(stats['rating_std']) else 0.0
            })

        # Convert numpy types to Python native types
        users = convert_numpy_types(users)

        return jsonify({
            "top_users": users,
            "count": len(users)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@collaborative_bp.route('/stats', methods=['GET'])
def system_stats():
    """
    获取系统统计信息
    """
    try:
        _, movies_df, ratings_df = get_cf_system()
        rec = get_cf_system()[0]

        # 基本统计
        n_users = ratings_df['user_id'].nunique()
        n_movies = movies_df['movie_id'].nunique()
        n_ratings = len(ratings_df)

        # 评分分布
        rating_dist = ratings_df['rating'].value_counts().sort_index()
        rating_distribution = []
        for rating, count in rating_dist.items():
            percentage = count / n_ratings * 100
            rating_distribution.append({
                "rating": convert_numpy_types(rating),
                "count": convert_numpy_types(int(count)),
                "percentage": convert_numpy_types(round(percentage, 1))
            })

        # 用户活跃度统计
        user_rating_counts = ratings_df['user_id'].value_counts()

        # 电影受欢迎度统计
        movie_rating_counts = ratings_df['movie_id'].value_counts()

        stats = {
            "basic_stats": {
                "users_count": convert_numpy_types(int(n_users)),
                "movies_count": convert_numpy_types(int(n_movies)),
                "ratings_count": convert_numpy_types(int(n_ratings)),
                "rating_density": convert_numpy_types(round(n_ratings / (n_users * n_movies) * 100, 2))
            },
            "rating_distribution": convert_numpy_types(rating_distribution),
            "user_activity": {
                "avg_ratings_per_user": convert_numpy_types(round(user_rating_counts.mean(), 1)),
                "max_ratings_per_user": convert_numpy_types(int(user_rating_counts.max())),
                "min_ratings_per_user": convert_numpy_types(int(user_rating_counts.min()))
            },
            "movie_popularity": {
                "avg_ratings_per_movie": convert_numpy_types(round(movie_rating_counts.mean(), 1)),
                "max_ratings_per_movie": convert_numpy_types(int(movie_rating_counts.max())),
                "min_ratings_per_movie": convert_numpy_types(int(movie_rating_counts.min()))
            },
            "model_config": {
                "n_factors": convert_numpy_types(rec.n_factors),
                "learning_rate": convert_numpy_types(rec.learning_rate),
                "regularization": convert_numpy_types(rec.regularization),
                "n_epochs": convert_numpy_types(rec.n_epochs)
            }
        }

        # Convert the final result to handle any numpy types
        stats = convert_numpy_types(stats)

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500