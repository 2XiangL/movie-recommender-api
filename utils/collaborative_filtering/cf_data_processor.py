"""
协同过滤推荐系统 - 数据处理模块
使用矩阵分解算法实现基于用户行为的推荐
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import ast
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ..cli.database_processor import DatabaseDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollaborativeDataProcessor:
    """协同过滤数据处理器"""

    def __init__(self, config):
        """
        初始化数据处理器

        Args:
            config: 配置字典
        """
        self.config = config

        # 检查是否使用数据库模式
        self.use_database = config.get("data", {}).get("use_database", False)
        self.db_processor = None

        if self.use_database:
            try:
                self.db_processor = DatabaseDataProcessor(config)
                logger.info("Database mode enabled for collaborative filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize database processor for CF, falling back to CSV: {e}")
                self.use_database = False

    def _get_data_path(self, filename: str) -> str:
        """获取数据文件完整路径"""
        data_dir = self.config["data"]["data_directory"]
        return os.path.join(data_dir, filename)

    def load_movie_data(self) -> pd.DataFrame:
        """加载电影数据"""
        logger.info("加载电影数据...")

        if self.use_database and self.db_processor:
            # 使用数据库模式
            logger.info("Using database mode for collaborative filtering...")
            movies_df = self.db_processor.load_movies_data_from_db()

            # 提取必要的电影信息
            available_columns = ['movie_id', 'title', 'vote_average', 'vote_count', 'popularity', 'genres']
            movies_processed = movies_df[available_columns].copy()

            # 数据库已经提供了列表格式的genres
            movies_processed['genres_list'] = movies_processed['genres'].apply(lambda x: x if isinstance(x, list) else [])
            movies_processed['main_genre'] = movies_processed['genres_list'].apply(lambda x: x[0] if x else 'Unknown')

        else:
            # 使用CSV模式
            logger.info("Using CSV mode for collaborative filtering...")
            movies_path = self._get_data_path(self.config["data"]["movies_csv"])

            if not os.path.exists(movies_path):
                raise FileNotFoundError(f"电影数据文件不存在: {movies_path}")

            movies_df = pd.read_csv(movies_path)

            # 提取必要的电影信息
            movies_processed = movies_df[['id', 'title', 'vote_average', 'vote_count', 'popularity', 'genres']].copy()
            movies_processed.rename(columns={'id': 'movie_id'}, inplace=True)

            # 解析电影类型
            def extract_genres(genres_str):
                try:
                    genres = ast.literal_eval(genres_str)
                    return [g['name'] for g in genres]
                except:
                    return []

            movies_processed['genres_list'] = movies_processed['genres'].apply(extract_genres)
            movies_processed['main_genre'] = movies_processed['genres_list'].apply(lambda x: x[0] if x else 'Unknown')

        logger.info(f"成功加载 {len(movies_processed)} 部电影数据")
        return movies_processed

    def generate_synthetic_user_data(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成模拟用户评分数据

        基于电影的流行度、评分和类型生成用户偏好
        """
        logger.info("生成模拟用户评分数据...")

        num_users = self.config["data"]["num_users"]
        min_ratings_per_user = self.config["data"]["min_ratings_per_user"]
        max_ratings_per_user = self.config["data"]["max_ratings_per_user"]

        # 电影特征权重
        popularity_weight = 0.4
        rating_weight = 0.6

        # 计算电影流行度分数
        movies_df['popularity_score'] = (
            movies_df['popularity'] / movies_df['popularity'].max() * popularity_weight +
            movies_df['vote_average'] / 10.0 * rating_weight
        )

        ratings_data = []

        for user_id in range(num_users):
            # 用户偏好随机种子
            np.random.seed(user_id + 42)  # 确保可重现

            # 每个用户偏好2-4个主要类型
            preferred_genres = np.random.choice(
                movies_df['main_genre'].unique(),
                size=np.random.randint(2, 5),
                replace=False
            )

            # 确定用户评分数量
            num_ratings = np.random.randint(min_ratings_per_user, max_ratings_per_user + 1)

            # 基于偏好选择电影
            preferred_movies = movies_df[movies_df['main_genre'].isin(preferred_genres)]

            if len(preferred_movies) < num_ratings:
                # 如果偏好类型电影不够，从其他类型补充
                other_movies = movies_df[~movies_df['main_genre'].isin(preferred_genres)]
                additional_needed = num_ratings - len(preferred_movies)
                other_movies_sample = other_movies.sample(
                    n=min(additional_needed, len(other_movies)),
                    weights='popularity_score'
                )
                selected_movies = pd.concat([preferred_movies, other_movies_sample])
            else:
                selected_movies = preferred_movies.sample(
                    n=num_ratings,
                    weights='popularity_score'
                )

            # 为每部电影生成评分
            for _, movie in selected_movies.iterrows():
                base_rating = movie['vote_average'] / 2.0  # 转换为5分制

                # 根据类型偏好调整评分
                if movie['main_genre'] in preferred_genres:
                    genre_bonus = np.random.uniform(0.5, 1.5)
                else:
                    genre_bonus = np.random.uniform(-0.5, 0.5)

                # 添加随机噪声
                noise = np.random.normal(0, 0.3)

                final_rating = base_rating + genre_bonus + noise
                final_rating = max(0.5, min(5.0, final_rating))  # 限制在0.5-5.0范围内

                ratings_data.append({
                    'user_id': user_id,
                    'movie_id': movie['movie_id'],
                    'rating': round(final_rating, 1),
                    'timestamp': np.random.randint(1000000000, 1700000000)  # 随机时间戳
                })

        ratings_df = pd.DataFrame(ratings_data)

        logger.info(f"成功生成 {len(ratings_df)} 条评分记录，涉及 {ratings_df['user_id'].nunique()} 个用户和 {ratings_df['movie_id'].nunique()} 部电影")

        return ratings_df

    def create_user_movie_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """创建用户-电影评分矩阵"""
        logger.info("创建用户-电影评分矩阵...")

        # 创建用户-电影评分矩阵
        user_movie_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )

        logger.info(f"评分矩阵维度: {user_movie_matrix.shape}")

        return user_movie_matrix

    def save_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                  user_movie_matrix: pd.DataFrame):
        """保存处理后的数据"""
        logger.info("保存处理后的数据...")

        data_dir = self.config["data"]["data_directory"]
        os.makedirs(data_dir, exist_ok=True)

        # 保存数据
        files_to_save = {
            'cf_movies_dict.pkl': movies_df,
            'cf_ratings_dict.pkl': ratings_df,
            'cf_user_movie_matrix.pkl': user_movie_matrix
        }

        for filename, data in files_to_save.items():
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(data.to_dict(), f)
            logger.info(f"数据已保存到: {filepath}")

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载处理后的数据"""
        logger.info("加载处理后的数据...")

        data_dir = self.config["data"]["data_directory"]

        files_to_load = {
            'movies': 'cf_movies_dict.pkl',
            'ratings': 'cf_ratings_dict.pkl',
            'user_movie_matrix': 'cf_user_movie_matrix.pkl'
        }

        data = {}
        for data_type, filename in files_to_load.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    loaded_dict = pickle.load(f)
                    data[data_type] = pd.DataFrame.from_dict(loaded_dict)
                logger.info(f"成功加载 {data_type} 数据: {data[data_type].shape}")
            else:
                raise FileNotFoundError(f"数据文件不存在: {filepath}")

        return data['movies'], data['ratings'], data['user_movie_matrix']