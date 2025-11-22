from flask import Blueprint, request, jsonify
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.cli.recommender import MovieRecommender
from utils.common import convert_numpy_types
from config.settings import Config

content_bp = Blueprint('content_based', __name__)

# 全局推荐器实例
recommender = None

def get_recommender():
    """获取推荐器实例（延迟加载）"""
    global recommender
    if recommender is None:
        try:
            recommender = MovieRecommender()
        except Exception as e:
            raise Exception(f"Failed to initialize content-based recommender: {e}")
    return recommender

@content_bp.route('/recommend', methods=['GET'])
def recommend_movies():
    """
    基于内容的电影推荐

    Query Parameters:
    - movie: 电影标题 (必需)
    - n: 推荐数量 (可选，默认为10)
    """
    try:
        movie_title = request.args.get('movie')
        if not movie_title:
            return jsonify({"error": "Missing required parameter: movie"}), 400

        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        rec = get_recommender()

        # Get original movie details first
        original_movie_details = rec.get_movie_details(movie_title)
        if not original_movie_details:
            suggestions = rec.search_movies(movie_title, limit=5)
            suggestions = convert_numpy_types(suggestions)
            return jsonify({
                "error": f"Movie not found: {movie_title}",
                "suggestions": suggestions
            }), 404

        # Get recommendations (excluding original movie)
        recommendations = rec.get_recommendations(movie_title, n)
        if not recommendations:
            # If no recommendations, return just the original movie
            original_movie = convert_numpy_types(original_movie_details)
            return jsonify({
                "movie": movie_title,
                "recommendations": original_movie,
                "count": 1
            })

        # Convert numpy types to Python native types
        recommendations = convert_numpy_types(recommendations)
        original_movie = convert_numpy_types(original_movie_details)

        # Add original movie as first recommendation with highest score
        all_recommendations = [
            {
                "title": original_movie.get("title", movie_title),
                "movie_id": original_movie.get("movie_id"),
                "avg_score": 1.0,
                "similarity_types": ["Original Movie"],
                **original_movie
            },
            *recommendations
        ]

        return jsonify({
            "movie": movie_title,
            "recommendations": all_recommendations[:n],  # Ensure we don't exceed requested limit
            "count": len(all_recommendations[:n])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@content_bp.route('/search', methods=['GET'])
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

        rec = get_recommender()
        results = rec.search_movies(query, limit=n)

        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@content_bp.route('/details', methods=['GET'])
def movie_details():
    """
    获取电影详细信息

    Query Parameters:
    - movie: 电影标题 (必需)
    """
    try:
        movie_title = request.args.get('movie')
        if not movie_title:
            return jsonify({"error": "Missing required parameter: movie"}), 400

        rec = get_recommender()
        details = rec.get_movie_details(movie_title)

        if not details:
            return jsonify({"error": f"Movie not found: {movie_title}"}), 404

        # Convert numpy types to Python native types
        details = convert_numpy_types(details)

        return jsonify({
            "movie": movie_title,
            "details": details
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@content_bp.route('/random', methods=['GET'])
def random_movies():
    """
    获取随机电影推荐

    Query Parameters:
    - n: 推荐数量 (可选，默认为10)
    """
    try:
        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        rec = get_recommender()
        random_movies = rec.get_random_movies(n)

        return jsonify({
            "random_movies": random_movies,
            "count": len(random_movies)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@content_bp.route('/similar', methods=['GET'])
def similar_movies():
    """
    查找相似电影（按特定相似度类型）

    Query Parameters:
    - movie: 电影标题 (必需)
    - type: 相似度类型 (可选，默认为tags)
      可选值: tags, genres, keywords, cast, production
    - n: 结果数量 (可选，默认为10)
    """
    try:
        movie_title = request.args.get('movie')
        if not movie_title:
            return jsonify({"error": "Missing required parameter: movie"}), 400

        similarity_type = request.args.get('type', 'tags')
        valid_types = ['tags', 'genres', 'keywords', 'cast', 'production']
        if similarity_type not in valid_types:
            return jsonify({"error": f"Invalid similarity type. Must be one of: {valid_types}"}), 400

        n = request.args.get('n', type=int, default=10)
        if n <= 0 or n > 50:
            return jsonify({"error": "Parameter n must be between 1 and 50"}), 400

        rec = get_recommender()
        similar = rec.find_similar_movies(movie_title, similarity_type)[:n]

        if not similar:
            return jsonify({"error": f"No similar movies found for: {movie_title}"}), 404

        return jsonify({
            "movie": movie_title,
            "similarity_type": similarity_type,
            "similar_movies": similar,
            "count": len(similar)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500