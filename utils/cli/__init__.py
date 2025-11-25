"""
Movie Recommendation System - CLI Package

This package contains the command-line interface components for the movie recommendation system.
"""

__version__ = "1.0.0"
__author__ = "Movie Recommender Team"

from .data_processor import DataProcessor
from .recommender import MovieRecommender

__all__ = ['DataProcessor', 'MovieRecommender']