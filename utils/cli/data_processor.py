import os
import json
import pickle
import string
import numpy as np
import pandas as pd
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict, List, Optional
import logging
from .database_processor import DatabaseDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing class for movie recommendation system."""

    def __init__(self, config):
        """
        Initialize the DataProcessor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ps = PorterStemmer()

        # 检查是否使用数据库模式
        self.use_database = config.get("data", {}).get("use_database", False)
        self.db_processor = None

        if self.use_database:
            try:
                self.db_processor = DatabaseDataProcessor(config)
                logger.info("Database mode enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize database processor, falling back to CSV: {e}")
                self.use_database = False

        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

    
    def _get_data_path(self, filename: str) -> str:
        """Get full path for data files."""
        data_dir = self.config["data"]["data_directory"]
        return os.path.join(data_dir, filename)

    def get_genres(self, obj: str) -> List[str]:
        """Extract genres from JSON string."""
        try:
            lista = ast.literal_eval(obj)
            return [i['name'] for i in lista]
        except (ValueError, KeyError):
            return []

    def get_cast(self, obj: str) -> List[str]:
        """Extract top 10 cast members from JSON string."""
        try:
            a = ast.literal_eval(obj)
            return [a[i]['name'] for i in range(min(10, len(a)))]
        except (ValueError, KeyError):
            return []

    def get_crew(self, obj: str) -> List[str]:
        """Extract directors from JSON string."""
        try:
            directors = []
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    directors.append(i['name'])
            return directors
        except (ValueError, KeyError):
            return []

    def stemming_stopwords(self, text_list: List[str]) -> str:
        """Apply stemming and remove stopwords from text list."""
        try:
            # Apply stemming
            stemmed = [self.ps.stem(word) for word in text_list]

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered = [word.lower() for word in stemmed if word.lower() not in stop_words and len(word) > 2]

            # Join and remove punctuation
            result = ' '.join(filtered)
            result = result.translate(str.maketrans('', '', string.punctuation))

            return result
        except Exception as e:
            logger.warning(f"Error in stemming_stopwords: {e}")
            return ''

    def load_and_process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and process the movie data from CSV files or database.

        Returns:
            Tuple of (movies_df, new_df, movies2_df)
        """
        logger.info("Loading and processing data...")

        if self.use_database and self.db_processor:
            # 使用数据库模式
            logger.info("Using database mode for data loading...")
            movies = self.db_processor.load_movies_data_from_db()

            # 重命名列以匹配CSV格式
            if 'id' in movies.columns:
                movies.rename(columns={'id': 'movie_id'}, inplace=True)

            # Create movies2 dataframe (simplified version)
            movies2 = movies.copy()
            available_columns = ['movie_id', 'title', 'budget', 'overview', 'popularity',
                               'release_date', 'revenue', 'runtime', 'status',
                               'vote_average', 'vote_count']
            movies2_columns = [col for col in available_columns if col in movies2.columns]
            movies2 = movies2[movies2_columns]

            # Extract features for recommendations
            feature_columns = ['movie_id', 'title', 'overview', 'genres', 'keywords',
                              'cast', 'crew', 'production_companies', 'release_date']
            available_feature_columns = [col for col in feature_columns if col in movies.columns]
            movies = movies[available_feature_columns]
            movies.dropna(subset=['title', 'movie_id'], inplace=True)

        else:
            # 使用CSV模式
            logger.info("Using CSV mode for data loading...")
            # Load CSV files
            movies_path = self._get_data_path(self.config["data"]["movies_csv"])
            credits_path = self._get_data_path(self.config["data"]["credits_csv"])

            if not os.path.exists(movies_path) or not os.path.exists(credits_path):
                raise FileNotFoundError("Data files not found. Please ensure CSV files are in the data directory.")

            movies = pd.read_csv(movies_path)
            credits = pd.read_csv(credits_path)

            # Merge dataframes
            movies = movies.merge(credits, on='title')

            # Create movies2 dataframe (simplified version)
            movies2 = movies.copy()
            movies2.drop(['homepage', 'tagline'], axis=1, inplace=True)
            movies2 = movies2[['movie_id', 'title', 'budget', 'overview', 'popularity',
                              'release_date', 'revenue', 'runtime', 'spoken_languages',
                              'status', 'vote_average', 'vote_count']]

            # Extract features for recommendations
            movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords',
                            'cast', 'crew', 'production_companies', 'release_date']]
            movies.dropna(inplace=True)

        # Apply feature extraction functions
        logger.info("Extracting features...")

        if self.use_database and self.db_processor:
            # 数据库模式 - 数据已经是列表格式，不需要解析JSON
            # 确保数据不为空
            movies['genres'] = movies['genres'].apply(lambda x: x if isinstance(x, list) else [])
            movies['keywords'] = movies['keywords'].apply(lambda x: x if isinstance(x, list) else [])
            # 从完整的cast列中提取前10个作为top_cast
            if 'cast' in movies.columns:
                movies['top_cast'] = movies['cast'].apply(lambda x: x[:10] if isinstance(x, list) and len(x) > 0 else [])
            else:
                movies['top_cast'] = []
            movies['director'] = movies['crew'].apply(lambda x: x if isinstance(x, list) else [])
            movies['production_comp'] = movies['production_companies'].apply(lambda x: x if isinstance(x, list) else [])
        else:
            # CSV模式 - 需要解析JSON字符串
            movies['genres'] = movies['genres'].apply(self.get_genres)
            movies['keywords'] = movies['keywords'].apply(self.get_genres)
            movies['top_cast'] = movies['cast'].apply(self.get_cast)
            movies['director'] = movies['crew'].apply(self.get_crew)
            movies['production_comp'] = movies['production_companies'].apply(self.get_genres)

        # Clean text data
        logger.info("Cleaning text data...")

        # 处理overview - 处理可能的None值
        movies['overview'] = movies['overview'].apply(lambda x: x.split() if pd.notna(x) and isinstance(x, str) else [])

        # 处理文本数据 - 处理可能的空列表
        movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x] if x else [])
        movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x] if x else [])
        movies['tcast'] = movies['top_cast'].apply(lambda x: [i.replace(" ", "") for i in x] if x else [])
        movies['tcrew'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x] if x else [])
        movies['tprduction_comp'] = movies['production_comp'].apply(lambda x: [i.replace(" ", "") for i in x] if x else [])

        # Create tags column
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tcast'] + movies['tcrew']

        # Create new dataframe for analysis
        new_df = movies[['movie_id', 'title', 'tags', 'genres', 'keywords', 'tcast', 'tcrew', 'tprduction_comp']]

        # Convert to strings
        new_df['genres'] = new_df['genres'].apply(lambda x: " ".join(x))
        new_df['tcast'] = new_df['tcast'].apply(lambda x: " ".join(x))
        new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: " ".join(x))

        new_df['tcast'] = new_df['tcast'].str.lower()
        new_df['genres'] = new_df['genres'].str.lower()
        new_df['tprduction_comp'] = new_df['tprduction_comp'].str.lower()

        # Apply stemming and stopwords removal
        logger.info("Applying NLP processing...")
        new_df['tags'] = new_df['tags'].apply(self.stemming_stopwords)
        new_df['keywords'] = new_df['keywords'].apply(self.stemming_stopwords)

        logger.info(f"Data processing complete. Processed {len(new_df)} movies.")
        return movies, new_df, movies2

    def save_processed_data(self, movies_df: pd.DataFrame, new_df: pd.DataFrame, movies2_df: pd.DataFrame):
        """Save processed dataframes to pickle files."""
        logger.info("Saving processed data...")

        data_dir = self.config["data"]["data_directory"]
        os.makedirs(data_dir, exist_ok=True)

        processed_files = self.config["data"]["processed_files"]

        # Map dataframe variables to config keys
        dataframes = {
            'movies_dict': movies_df,
            'new_df_dict': new_df,
            'movies2_dict': movies2_df
        }

        # Save dataframes
        for df_name, filename in processed_files.items():
            if df_name in dataframes:
                df_dict = dataframes[df_name].to_dict()
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(df_dict, f)
                logger.info(f"Saved {df_name} to {filepath}")
            else:
                logger.warning(f"Dataframe {df_name} not found for saving")

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed dataframes from pickle files."""
        logger.info("Loading processed data...")

        data_dir = self.config["data"]["data_directory"]
        processed_files = self.config["data"]["processed_files"]

        dataframes = {}
        for df_name, filename in processed_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    loaded_dict = pickle.load(f)
                    dataframes[df_name.replace('_dict', '')] = pd.DataFrame.from_dict(loaded_dict)
                logger.info(f"Loaded {df_name} from {filepath}")
            else:
                raise FileNotFoundError(f"Processed data file {filepath} not found. Please run training first.")

        return dataframes['movies'], dataframes['new_df'], dataframes['movies2']

    def create_similarity_matrices(self, new_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create similarity matrices for different features."""
        logger.info("Creating similarity matrices...")

        similarity_types = self.config["recommendation"]["similarity_types"]
        max_features = self.config["model"]["vectorizer_max_features"]

        similarity_matrices = {}

        for sim_type in similarity_types:
            logger.info(f"Creating similarity matrix for {sim_type}...")

            cv = CountVectorizer(max_features=max_features, stop_words='english')
            try:
                if sim_type in new_df.columns:
                    vec_data = cv.fit_transform(new_df[sim_type]).toarray()
                    similarity_matrix = cosine_similarity(vec_data)
                    similarity_matrices[sim_type] = similarity_matrix
                    logger.info(f"Created {sim_type} similarity matrix: {similarity_matrix.shape}")
                else:
                    logger.warning(f"Column {sim_type} not found in dataframe")
            except Exception as e:
                logger.error(f"Error creating similarity matrix for {sim_type}: {e}")

        return similarity_matrices

    def save_similarity_matrices(self, similarity_matrices: Dict[str, np.ndarray]):
        """Save similarity matrices to pickle files."""
        logger.info("Saving similarity matrices...")

        data_dir = self.config["data"]["data_directory"]
        similarity_files = self.config["data"]["similarity_files"]

        for sim_type, matrix in similarity_matrices.items():
            filename = similarity_files.get(sim_type, f"similarity_{sim_type}.pkl")
            filepath = os.path.join(data_dir, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(matrix, f)
            logger.info(f"Saved {sim_type} similarity matrix to {filepath}")

    def load_similarity_matrices(self) -> Dict[str, np.ndarray]:
        """Load similarity matrices from pickle files."""
        logger.info("Loading similarity matrices...")

        data_dir = self.config["data"]["data_directory"]
        similarity_files = self.config["data"]["similarity_files"]

        similarity_matrices = {}

        for sim_type, filename in similarity_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    similarity_matrices[sim_type] = pickle.load(f)
                logger.info(f"Loaded {sim_type} similarity matrix: {similarity_matrices[sim_type].shape}")
            else:
                logger.warning(f"Similarity matrix file {filepath} not found")

        if not similarity_matrices:
            raise FileNotFoundError("No similarity matrices found. Please run training first.")

        return similarity_matrices