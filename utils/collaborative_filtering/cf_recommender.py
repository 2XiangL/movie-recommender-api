"""
协同过滤推荐算法核心实现
使用矩阵分解(Matrix Factorization)算法
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Tuple, List, Dict, Optional
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender:
    """协同过滤推荐引擎"""

    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01,
                 regularization: float = 0.1, n_epochs: int = 100):
        """
        初始化推荐引擎

        Args:
            n_factors: 潜在因子数量
            learning_rate: 学习率
            regularization: 正则化参数
            n_epochs: 训练轮数
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs

        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None

        self.user_encoder = None
        self.item_encoder = None
        self.movie_info = None

    def _create_mappings(self, ratings_df: pd.DataFrame):
        """创建用户和电影的映射关系"""
        # 创建用户和电影的编码器
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(ratings_df['user_id'].unique())}
        self.item_encoder = {movie_id: idx for idx, movie_id in enumerate(ratings_df['movie_id'].unique())}

        # 创建反向映射
        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        self.item_decoder = {idx: movie_id for movie_id, idx in self.item_encoder.items()}

    def fit(self, ratings_df: pd.DataFrame, movie_info_df: pd.DataFrame):
        """
        训练协同过滤模型

        Args:
            ratings_df: 评分数据 DataFrame
            movie_info_df: 电影信息 DataFrame
        """
        logger.info("开始训练协同过滤模型...")

        self.movie_info = movie_info_df.set_index('movie_id')['title'].to_dict()
        self._create_mappings(ratings_df)

        # 准备训练数据
        n_users = len(self.user_encoder)
        n_items = len(self.item_encoder)

        # 初始化参数
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()

        logger.info(f"用户数量: {n_users}, 电影数量: {n_items}")
        logger.info(f"全局平均评分: {self.global_mean:.2f}")

        # 将用户ID和电影ID转换为索引
        user_indices = ratings_df['user_id'].map(self.user_encoder).values
        item_indices = ratings_df['movie_id'].map(self.item_encoder).values
        ratings = ratings_df['rating'].values

        # 训练模型
        logger.info("开始矩阵分解训练...")
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            permutation = np.random.permutation(len(ratings))

            epoch_loss = 0
            for idx in permutation:
                user_idx = user_indices[idx]
                item_idx = item_indices[idx]
                rating = ratings[idx]

                # 预测评分
                prediction = (self.global_mean +
                            self.user_biases[user_idx] +
                            self.item_biases[item_idx] +
                            np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

                # 计算误差
                error = rating - prediction

                # 更新参数 (随机梯度下降)
                user_factor_old = self.user_factors[user_idx].copy()
                item_factor_old = self.item_factors[item_idx].copy()

                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_factor_old - self.regularization * user_factor_old
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factor_old - self.regularization * item_factor_old
                )

                self.user_biases[user_idx] += self.learning_rate * (
                    error - self.regularization * self.user_biases[user_idx]
                )
                self.item_biases[item_idx] += self.learning_rate * (
                    error - self.regularization * self.item_biases[item_idx]
                )

                epoch_loss += error ** 2

            # 计算RMSE
            rmse = np.sqrt(epoch_loss / len(ratings))

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

        logger.info("模型训练完成!")

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        预测用户对电影的评分

        Args:
            user_id: 用户ID
            movie_id: 电影ID

        Returns:
            预测评分
        """
        if user_id not in self.user_encoder or movie_id not in self.item_encoder:
            return self.global_mean

        user_idx = self.user_encoder[user_id]
        item_idx = self.item_encoder[movie_id]

        prediction = (self.global_mean +
                     self.user_biases[user_idx] +
                     self.item_biases[item_idx] +
                     np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

        return max(0.5, min(5.0, prediction))  # 限制评分范围

    def recommend_for_user(self, user_id: int, n_recommendations: int = 10,
                          exclude_seen: bool = True, ratings_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        为用户推荐电影

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            exclude_seen: 是否排除用户已评分的电影
            ratings_df: 评分数据，用于排除已评分电影

        Returns:
            推荐电影列表
        """
        if user_id not in self.user_encoder:
            logger.warning(f"用户 {user_id} 不在训练数据中，返回热门电影")
            return []

        user_idx = self.user_encoder[user_id]
        seen_movies = set()

        if exclude_seen and ratings_df is not None:
            seen_movies = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist())

        # 计算用户对所有未评分电影的预测评分
        recommendations = []
        user_vector = self.user_factors[user_idx]
        user_bias = self.user_biases[user_idx]

        for movie_idx in range(len(self.item_decoder)):
            movie_id = self.item_decoder[movie_idx]

            if exclude_seen and movie_id in seen_movies:
                continue

            # 计算预测评分
            prediction = (self.global_mean +
                         user_bias +
                         self.item_biases[movie_idx] +
                         np.dot(user_vector, self.item_factors[movie_idx]))

            movie_title = self.movie_info.get(movie_id, f"Movie {movie_id}")

            recommendations.append({
                'movie_id': movie_id,
                'title': movie_title,
                'predicted_rating': max(0.5, min(5.0, prediction))
            })

        # 按预测评分排序
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)

        return recommendations[:n_recommendations]

    def find_similar_users(self, user_id: int, n_similar: int = 10) -> List[Dict]:
        """
        找到相似用户

        Args:
            user_id: 目标用户ID
            n_similar: 相似用户数量

        Returns:
            相似用户列表
        """
        if user_id not in self.user_encoder:
            return []

        user_idx = self.user_encoder[user_id]
        user_vector = self.user_factors[user_idx].reshape(1, -1)

        # 计算与所有用户的相似度
        similarities = cosine_similarity(user_vector, self.user_factors)[0]

        # 获取最相似的用户 (排除自己)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]

        similar_users = []
        for idx in similar_indices:
            similar_user_id = self.user_decoder[idx]
            similar_users.append({
                'user_id': similar_user_id,
                'similarity': similarities[idx]
            })

        return similar_users

    def find_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Dict]:
        """
        找到相似电影 (基于物品协同过滤)

        Args:
            movie_id: 目标电影ID
            n_similar: 相似电影数量

        Returns:
            相似电影列表
        """
        if movie_id not in self.item_encoder:
            return []

        movie_idx = self.item_encoder[movie_id]
        movie_vector = self.item_factors[movie_idx].reshape(1, -1)

        # 计算与所有电影的相似度
        similarities = cosine_similarity(movie_vector, self.item_factors)[0]

        # 获取最相似的电影 (排除自己)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]

        similar_movies = []
        for idx in similar_indices:
            similar_movie_id = self.item_decoder[idx]
            similar_movies.append({
                'movie_id': similar_movie_id,
                'title': self.movie_info.get(similar_movie_id, f"Movie {similar_movie_id}"),
                'similarity': similarities[idx]
            })

        return similar_movies

    def get_user_profile(self, user_id: int, ratings_df: pd.DataFrame) -> Dict:
        """
        获取用户画像信息

        Args:
            user_id: 用户ID
            ratings_df: 评分数据

        Returns:
            用户画像
        """
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]

        if len(user_ratings) == 0:
            return {'error': f'用户 {user_id} 没有评分记录'}

        # 基本统计信息
        profile = {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_std': user_ratings['rating'].std(),
            'favorite_genres': [],
            'highly_rated_movies': []
        }

        # 获取用户高评分电影
        high_rated = user_ratings[user_ratings['rating'] >= 4.0]
        if len(high_rated) > 0:
            profile['highly_rated_movies'] = high_rated.nlargest(5, 'rating')[
                ['movie_id', 'rating']
            ].to_dict('records')

        return profile

    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'global_mean': self.global_mean,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_decoder': self.user_decoder,
            'item_decoder': self.item_decoder,
            'movie_info': self.movie_info,
            'config': {
                'n_factors': self.n_factors,
                'learning_rate': self.learning_rate,
                'regularization': self.regularization,
                'n_epochs': self.n_epochs
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.user_biases = model_data['user_biases']
        self.item_biases = model_data['item_biases']
        self.global_mean = model_data['global_mean']
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.user_decoder = model_data['user_decoder']
        self.item_decoder = model_data['item_decoder']
        self.movie_info = model_data['movie_info']

        config = model_data['config']
        self.n_factors = config['n_factors']
        self.learning_rate = config['learning_rate']
        self.regularization = config['regularization']
        self.n_epochs = config['n_epochs']

        logger.info(f"模型已从 {filepath} 加载")

    def evaluate(self, test_ratings: pd.DataFrame) -> Dict:
        """
        评估模型性能

        Args:
            test_ratings: 测试评分数据

        Returns:
            评估结果
        """
        predictions = []
        actuals = []

        for _, row in test_ratings.iterrows():
            pred = self.predict(row['user_id'], row['movie_id'])
            predictions.append(pred)
            actuals.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions)
        }