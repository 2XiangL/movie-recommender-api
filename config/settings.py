import os
from pathlib import Path

class Config:
    """Flask应用配置"""

    # 基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # 数据目录
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'

    # 内容推荐配置
    CONTENT_CONFIG = {
        "data": {
            "data_directory": str(DATA_DIR),
            "movies_csv": "tmdb_5000_movies.csv",
            "credits_csv": "tmdb_5000_credits.csv",
            "use_database": True,  # 设置为True以启用数据库模式
            "database_path": "movies.db",  # SQLite数据库文件路径
            "processed_files": {
                "movies_dict": "movies_dict.pkl",
                "new_df_dict": "new_df_dict.pkl",
                "movies2_dict": "movies2_dict.pkl"
            },
            "similarity_files": {
                "tags": "similarity_tags_tags.pkl",
                "genres": "similarity_tags_genres.pkl",
                "keywords": "similarity_tags_keywords.pkl",
                "cast": "similarity_tcast.pkl",
                "production": "similarity_tprduction_comp.pkl"
            }
        },
        "model": {
            "vectorizer_max_features": 5000,
            "min_similarity_score": 0.1
        },
        "recommendation": {
            "max_recommendations": 20,
            "similarity_threshold": 0.1,
            "similarity_types": ["tags", "genres", "keywords", "cast", "production"]
        },
        "display": {
            "max_title_length": 50,
            "show_similarity_scores": True,
            "show_movie_year": True
        }
    }

    # 协同过滤配置
    COLLABORATIVE_CONFIG = {
        "data": {
            "data_directory": str(DATA_DIR),
            "movies_csv": "tmdb_5000_movies.csv",
            "use_database": True,  # 设置为True以启用数据库模式
            "database_path": "movies.db",  # SQLite数据库文件路径
            "num_users": 1000,
            "min_ratings_per_user": 20,
            "max_ratings_per_user": 100
        },
        "model": {
            "n_factors": 50,
            "learning_rate": 0.01,
            "regularization": 0.1,
            "n_epochs": 100,
            "test_size": 0.2
        },
        "recommendation": {
            "default_n_recommendations": 10,
            "min_rating_threshold": 3.5,
            "similarity_threshold": 0.1
        },
        "display": {
            "max_title_length": 50
        },
        "files": {
            "cf_model": "cf_model.pkl",
            "user_ratings": "cf_ratings_dict.pkl"
        }
    }
