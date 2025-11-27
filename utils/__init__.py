"""
协同过滤推荐系统包
"""

__version__ = "1.0.0"
__author__ = "Movie Recommender Team"

from .cf_data_processor import CollaborativeDataProcessor
from .cf_recommender import CollaborativeFilteringRecommender

__all__ = ['CollaborativeDataProcessor', 'CollaborativeFilteringRecommender']