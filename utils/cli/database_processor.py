"""
数据库数据处理器 - 用于替代CSV读取
使用SQLite数据库提供高效的数据查询和处理
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import ast
import logging
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseDataProcessor:
    """基于SQLite数据库的数据处理器"""

    def __init__(self, config: Dict):
        """
        初始化数据库数据处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.db_path = config.get("data", {}).get("database_path", "movies.db")

        # 如果没有指定完整路径，则相对于项目根目录
        if not os.path.isabs(self.db_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.db_path = os.path.join(base_dir, self.db_path)

        logger.info(f"Database path: {self.db_path}")

        # 验证数据库文件存在
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 启用行工厂，便于按列名访问
        try:
            yield conn
        finally:
            conn.close()

    def get_genres_from_db(self, movie_id: int) -> List[str]:
        """从数据库获取电影类型"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT g.name
                    FROM genres g
                    JOIN movies_genres mg ON g.id = mg.genre_id
                    WHERE mg.movie_id = ?
                    ORDER BY g.name
                """, (movie_id,))
                return [row['name'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching genres for movie {movie_id}: {e}")
            return []

    def get_keywords_from_db(self, movie_id: int) -> List[str]:
        """从数据库获取电影关键词"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT k.name
                    FROM keywords k
                    JOIN movies_keywords mk ON k.id = mk.keyword_id
                    WHERE mk.movie_id = ?
                    ORDER BY k.name
                """, (movie_id,))
                return [row['name'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching keywords for movie {movie_id}: {e}")
            return []

    def get_cast_from_db(self, movie_id: int, limit: int = 10) -> List[str]:
        """从数据库获取电影演员（前limit名）"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.name
                    FROM people p
                    JOIN movies_cast mc ON p.id = mc.person_id
                    WHERE mc.movie_id = ?
                    ORDER BY mc.order_num ASC
                    LIMIT ?
                """, (movie_id, limit))
                return [row['name'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching cast for movie {movie_id}: {e}")
            return []

    def get_crew_from_db(self, movie_id: int, job_filter: str = 'Director') -> List[str]:
        """从数据库获取电影工作人员（按职位筛选）"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.name
                    FROM people p
                    JOIN movies_crew mc ON p.id = mc.person_id
                    WHERE mc.movie_id = ? AND mc.job = ?
                    ORDER BY p.name
                """, (movie_id, job_filter))
                return [row['name'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching crew for movie {movie_id}: {e}")
            return []

    def get_production_companies_from_db(self, movie_id: int) -> List[str]:
        """从数据库获取制作公司"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT pc.name
                    FROM production_companies pc
                    JOIN movies_production_companies mpc ON pc.id = mpc.company_id
                    WHERE mpc.movie_id = ?
                    ORDER BY pc.name
                """, (movie_id,))
                return [row['name'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching production companies for movie {movie_id}: {e}")
            return []

    def load_movies_data_from_db(self) -> pd.DataFrame:
        """从数据库加载电影数据"""
        logger.info("Loading movie data from database...")

        try:
            with self.get_connection() as conn:
                # 获取基本电影信息
                movies_df = pd.read_sql_query("""
                    SELECT
                        id as movie_id,
                        title,
                        overview,
                        budget,
                        popularity,
                        release_date,
                        revenue,
                        runtime,
                        status,
                        vote_average,
                        vote_count,
                        original_language,
                        original_title,
                        tagline
                    FROM movies
                    ORDER BY title
                """, conn)

                logger.info(f"Loaded {len(movies_df)} movies from database")

                # 添加额外的特征列
                movies_df['genres'] = movies_df['movie_id'].apply(self.get_genres_from_db)
                movies_df['keywords'] = movies_df['movie_id'].apply(self.get_keywords_from_db)
                movies_df['cast'] = movies_df['movie_id'].apply(lambda x: self.get_cast_from_db(x, 100))
                movies_df['top_cast'] = movies_df['movie_id'].apply(lambda x: self.get_cast_from_db(x, 10))
                movies_df['crew'] = movies_df['movie_id'].apply(lambda x: self.get_crew_from_db(x, 'Director'))
                movies_df['production_companies'] = movies_df['movie_id'].apply(self.get_production_companies_from_db)

                return movies_df

        except Exception as e:
            logger.error(f"Error loading movies from database: {e}")
            raise

    def get_movie_by_title(self, title: str) -> Optional[Dict]:
        """根据标题查找电影"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        id as movie_id,
                        title,
                        overview,
                        budget,
                        popularity,
                        release_date,
                        revenue,
                        runtime,
                        status,
                        vote_average,
                        vote_count,
                        original_language,
                        original_title,
                        tagline
                    FROM movies
                    WHERE title = ?
                """, (title,))
                row = cursor.fetchone()

                if row:
                    movie_dict = dict(row)
                    # 添加关联数据
                    movie_dict['genres'] = self.get_genres_from_db(movie_dict['movie_id'])
                    movie_dict['keywords'] = self.get_keywords_from_db(movie_dict['movie_id'])
                    movie_dict['cast'] = self.get_cast_from_db(movie_dict['movie_id'], 100)
                    movie_dict['top_cast'] = self.get_cast_from_db(movie_dict['movie_id'], 10)
                    movie_dict['crew'] = self.get_crew_from_db(movie_dict['movie_id'], 'Director')
                    movie_dict['production_companies'] = self.get_production_companies_from_db(movie_dict['movie_id'])
                    return movie_dict
                return None

        except Exception as e:
            logger.error(f"Error finding movie by title '{title}': {e}")
            return None

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """根据ID查找电影"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        id as movie_id,
                        title,
                        overview,
                        budget,
                        popularity,
                        release_date,
                        revenue,
                        runtime,
                        status,
                        vote_average,
                        vote_count,
                        original_language,
                        original_title,
                        tagline
                    FROM movies
                    WHERE id = ?
                """, (movie_id,))
                row = cursor.fetchone()

                if row:
                    movie_dict = dict(row)
                    # 添加关联数据
                    movie_dict['genres'] = self.get_genres_from_db(movie_dict['movie_id'])
                    movie_dict['keywords'] = self.get_keywords_from_db(movie_dict['movie_id'])
                    movie_dict['cast'] = self.get_cast_from_db(movie_dict['movie_id'], 100)
                    movie_dict['top_cast'] = self.get_cast_from_db(movie_dict['movie_id'], 10)
                    movie_dict['crew'] = self.get_crew_from_db(movie_dict['movie_id'], 'Director')
                    movie_dict['production_companies'] = self.get_production_companies_from_db(movie_dict['movie_id'])
                    return movie_dict
                return None

        except Exception as e:
            logger.error(f"Error finding movie by ID {movie_id}: {e}")
            return None

    def search_movies_by_title(self, query: str, limit: int = 10) -> List[Dict]:
        """根据标题搜索电影（模糊匹配）"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        id as movie_id,
                        title,
                        overview,
                        vote_average,
                        vote_count,
                        popularity
                    FROM movies
                    WHERE title LIKE ?
                    ORDER BY
                        CASE
                            WHEN title = ? THEN 1
                            WHEN title LIKE ? || '%' THEN 2
                            WHEN title LIKE '%' || ? || '%' THEN 3
                            ELSE 4
                        END,
                        popularity DESC,
                        title
                    LIMIT ?
                """, (f'%{query}%', query, query, query, limit))

                results = []
                for row in cursor.fetchall():
                    movie_dict = dict(row)
                    results.append(movie_dict)

                return results

        except Exception as e:
            logger.error(f"Error searching movies by title '{query}': {e}")
            return []

    def get_random_movies(self, limit: int = 10) -> List[Dict]:
        """获取随机电影"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        id as movie_id,
                        title,
                        overview,
                        vote_average,
                        vote_count,
                        popularity
                    FROM movies
                    ORDER BY RANDOM()
                    LIMIT ?
                """, (limit,))

                results = []
                for row in cursor.fetchall():
                    movie_dict = dict(row)
                    results.append(movie_dict)

                return results

        except Exception as e:
            logger.error(f"Error getting random movies: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # 获取各表的记录数
                tables = ['movies', 'genres', 'keywords', 'people', 'production_companies',
                         'movies_genres', 'movies_keywords', 'movies_cast', 'movies_crew']

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                # 获取电影评分统计
                cursor.execute("SELECT AVG(vote_average), MIN(vote_average), MAX(vote_average) FROM movies WHERE vote_average > 0")
                avg_vote, min_vote, max_vote = cursor.fetchone()
                stats['vote_stats'] = {
                    'average': avg_vote,
                    'min': min_vote,
                    'max': max_vote
                }

                # 获取类型统计
                cursor.execute("""
                    SELECT g.name, COUNT(mg.movie_id) as movie_count
                    FROM genres g
                    JOIN movies_genres mg ON g.id = mg.genre_id
                    GROUP BY g.id, g.name
                    ORDER BY movie_count DESC
                """)
                stats['genre_distribution'] = dict(cursor.fetchall())

                return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}