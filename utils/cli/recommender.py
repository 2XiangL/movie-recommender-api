import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from utils.cli.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class MovieRecommender:
    """Movie recommendation engine."""

    def __init__(self, config=None):
        """
        Initialize the MovieRecommender.

        Args:
            config: Configuration dictionary
        """
        from config.settings import Config
        if config is None:
            config = Config.CONTENT_CONFIG
        self.data_processor = DataProcessor(config)
        self.config = self.data_processor.config
        self.new_df = None
        self.movies_df = None
        self.movies2_df = None
        self.similarity_matrices = {}
        self._load_data()

    def _load_data(self):
        """Load all required data."""
        try:
            self.movies_df, self.new_df, self.movies2_df = self.data_processor.load_processed_data()
            self.similarity_matrices = self.data_processor.load_similarity_matrices()
            logger.info("All data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_movie_info(self, movie_title: str) -> Optional[pd.Series]:
        """
        Get detailed information about a movie.

        Args:
            movie_title: Title of the movie

        Returns:
            Series with movie information or None if not found
        """
        try:
            movie_row = self.new_df[self.new_df['title'].str.lower() == movie_title.lower()]
            if movie_row.empty:
                # Try partial match
                movie_row = self.new_df[self.new_df['title'].str.contains(movie_title, case=False, na=False)]

            if not movie_row.empty:
                return movie_row.iloc[0]
            return None
        except Exception as e:
            logger.error(f"Error getting movie info: {e}")
            return None

    def find_similar_movies(self, movie_title: str, similarity_type: str = "tags") -> List[Dict]:
        """
        Find similar movies based on a similarity type.

        Args:
            movie_title: Title of the movie to find recommendations for
            similarity_type: Type of similarity to use ("tags", "genres", "keywords", "cast", "production")

        Returns:
            List of dictionaries with movie info and similarity scores
        """
        try:
            if similarity_type not in self.similarity_matrices:
                logger.error(f"Similarity type '{similarity_type}' not available")
                return []

            similarity_matrix = self.similarity_matrices[similarity_type]

            # Find the movie index
            movie_info = self.get_movie_info(movie_title)
            if movie_info is None:
                logger.warning(f"Movie '{movie_title}' not found")
                return []

            movie_idx = movie_info.name  # Get the index of the movie

            # Get similarity scores
            similarity_scores = similarity_matrix[movie_idx]

            # Create list of (index, score) tuples
            movie_scores = list(enumerate(similarity_scores))

            # Sort by similarity score (descending) and exclude the movie itself
            movie_scores.sort(key=lambda x: x[1], reverse=True)

            # Filter out the input movie and apply threshold
            threshold = self.config["model"]["min_similarity_score"]
            recommendations = []

            for idx, score in movie_scores[1:]:  # Skip first (it's the movie itself)
                if score >= threshold:
                    movie_row = self.new_df.iloc[idx]
                    movie_name = movie_row['title']
                    movie_id = movie_row['movie_id']
                    recommendations.append({
                        'movie_id': int(movie_id),
                        'title': movie_name,
                        'similarity': float(score)
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Error finding similar movies: {e}")
            return []

    def get_recommendations(self, movie_title: str, num_recommendations: Optional[int] = None) -> List[Dict]:
        """
        Get comprehensive recommendations for a movie.

        Args:
            movie_title: Title of the movie
            num_recommendations: Number of recommendations to return (from config if None)

        Returns:
            List of recommendation dictionaries with movie info and scores
        """
        if num_recommendations is None:
            num_recommendations = self.config["recommendation"]["max_recommendations"]

        logger.info(f"Getting {num_recommendations} recommendations for '{movie_title}'")

        # Check if movie exists
        movie_info = self.get_movie_info(movie_title)
        if movie_info is None:
            logger.error(f"Movie '{movie_title}' not found in database")
            return []

        similarity_types = self.config["recommendation"]["similarity_types"]
        all_recommendations = {}
        threshold = self.config["recommendation"]["similarity_threshold"]

        # Get recommendations from all similarity types
        for sim_type in similarity_types:
            if sim_type in self.similarity_matrices:
                recommendations = self.find_similar_movies(movie_title, sim_type)
                sim_type_name = self._get_similarity_type_name(sim_type)

                for movie_data in recommendations:
                    movie_name = movie_data['title']
                    score = movie_data['similarity']
                    movie_id = movie_data['movie_id']

                    if score >= threshold:
                        if movie_name not in all_recommendations:
                            all_recommendations[movie_name] = {
                                'movie_id': movie_id,
                                'title': movie_name,
                                'scores': {},
                                'avg_score': 0.0,
                                'similarity_types': []
                            }

                        all_recommendations[movie_name]['scores'][sim_type_name] = score
                        all_recommendations[movie_name]['similarity_types'].append(sim_type_name)

        # Calculate average scores
        for movie_name, rec_data in all_recommendations.items():
            scores = rec_data['scores'].values()
            rec_data['avg_score'] = sum(scores) / len(scores)

        # Sort by average score and limit recommendations
        sorted_recommendations = sorted(
            all_recommendations.values(),
            key=lambda x: x['avg_score'],
            reverse=True
        )

        return sorted_recommendations[:num_recommendations]

    def _get_similarity_type_name(self, sim_type: str) -> str:
        """Get display name for similarity type."""
        type_names = {
            'tags': 'Content Similarity',
            'genres': 'Genre Match',
            'keywords': 'Keywords Match',
            'tcast': 'Cast Similarity',
            'tprduction_comp': 'Production Company'
        }
        return type_names.get(sim_type, sim_type.title())

    def get_movie_details(self, movie_title: str) -> Optional[Dict]:
        """
        Get detailed information about a movie.

        Args:
            movie_title: Title of the movie

        Returns:
            Dictionary with movie details or None if not found
        """
        try:
            movie_info = self.get_movie_info(movie_title)
            if movie_info is None:
                return None

            # Get additional details from movies2_df dataframe
            movie_row = self.movies2_df[self.movies2_df['title'].str.lower() == movie_title.lower()]
            if movie_row.empty:
                movie_row = self.movies2_df[self.movies2_df['title'].str.contains(movie_title, case=False, na=False)]

            details = {
                'title': movie_info['title'],
                'movie_id': movie_info['movie_id'],
                'tags': movie_info['tags'],
                'genres': movie_info['genres'],
                'keywords': movie_info['keywords'],
                'cast': movie_info['tcast'],
                'director': movie_info['tcrew'],
                'production_company': movie_info['tprduction_comp']
            }

            # Add additional details if available
            if not movie_row.empty:
                row = movie_row.iloc[0]
                details.update({
                    'budget': row.get('budget', 'N/A'),
                    'revenue': row.get('revenue', 'N/A'),
                    'runtime': row.get('runtime', 'N/A'),
                    'vote_average': row.get('vote_average', 'N/A'),
                    'vote_count': row.get('vote_count', 'N/A'),
                    'release_date': row.get('release_date', 'N/A'),
                    'overview': row.get('overview', 'N/A')
                })

            return details

        except Exception as e:
            logger.error(f"Error getting movie details: {e}")
            return None

    def search_movies(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for movies by title.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of dictionaries with movie info
        """
        try:
            # Convert dictionary to DataFrame for easier searching
            if isinstance(self.new_df, dict):
                import pandas as pd
                new_df = pd.DataFrame(self.new_df)
            else:
                new_df = self.new_df

            # Search for movies containing the query
            matches = new_df[
                new_df['title'].str.contains(query, case=False, na=False)
            ]

            if not matches.empty:
                results = []
                for _, row in matches.head(limit).iterrows():
                    results.append({
                        'movie_id': int(row['movie_id']),
                        'title': row['title']
                    })
                return results
            else:
                # Try fuzzy match using string similarity
                from difflib import get_close_matches
                all_titles = new_df['title'].tolist()
                title_matches = get_close_matches(query, all_titles, n=limit, cutoff=0.6)

                results = []
                for title in title_matches:
                    movie_row = new_df[new_df['title'] == title].iloc[0]
                    results.append({
                        'movie_id': int(movie_row['movie_id']),
                        'title': title
                    })
                return results

        except Exception as e:
            logger.error(f"Error searching movies: {e}")
            return []

    def get_random_movies(self, limit: int = 10) -> List[Dict]:
        """
        Get random movie suggestions.

        Args:
            limit: Number of random movies to return

        Returns:
            List of dictionaries with random movie info
        """
        try:
            # Convert dictionary to DataFrame for easier sampling
            if isinstance(self.new_df, dict):
                import pandas as pd
                new_df = pd.DataFrame(self.new_df)
            else:
                new_df = self.new_df

            random_movies = new_df.sample(limit)
            results = []
            for _, row in random_movies.iterrows():
                results.append({
                    'movie_id': int(row['movie_id']),
                    'title': row['title']
                })
            return results
        except Exception as e:
            logger.error(f"Error getting random movies: {e}")
            return []