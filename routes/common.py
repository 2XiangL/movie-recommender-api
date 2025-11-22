from flask import Blueprint, jsonify
import os
from pathlib import Path

common_bp = Blueprint('common', __name__)

@common_bp.route('/info', methods=['GET'])
def system_info():
    """获取系统信息"""
    try:
        base_dir = Path(__file__).parent.parent

        # 检查数据文件
        data_dir = base_dir / 'data'
        movies_csv = data_dir / 'tmdb_5000_movies.csv'
        credits_csv = data_dir / 'tmdb_5000_credits.csv'

        # 检查模型文件
        cf_model = data_dir / 'cf_model.pkl'
        similarity_files = list(data_dir.glob('similarity_*.pkl'))

        info = {
            "system": "Movie Recommendation API",
            "version": "1.0.0",
            "endpoints": {
                "content_based": "/api/content-based/*",
                "collaborative": "/api/collaborative/*",
                "common": "/api/*"
            },
            "data_status": {
                "movies_csv": str(movies_csv.exists()),
                "credits_csv": str(credits_csv.exists()),
                "cf_model": str(cf_model.exists()),
                "similarity_matrices": len(similarity_files)
            },
            "data_directory": str(data_dir),
            "available_methods": [
                "Content-based filtering",
                "Collaborative filtering (Matrix Factorization)"
            ]
        }

        return jsonify(info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@common_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "message": "Movie Recommendation API is running"
    })