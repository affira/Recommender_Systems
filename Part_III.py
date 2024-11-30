import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')

class DiverseGroupRecommender:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.user_movie_matrix = None
        self.user_means = None
        self.genre_matrix = None
        self.user_similarity_cache = {}  # Cache for user similarities
        self.item_diversity_cache = {}   # Cache for item diversities

    def load_data(self, data_path: str = 'u.data', item_path: str = 'u.item') -> None:
        # Load ratings
        self.ratings_df = pd.read_csv(
            data_path,
            sep='\t',
            names=['userId', 'movieId', 'rating', 'timestamp'])

        # Load movies with genre information
        columns = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url']
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        columns.extend(genre_columns)

        self.movies_df = pd.read_csv(item_path, sep='|', names=columns, encoding='latin-1')
        
        # Optimize matrix creation
        self.user_movie_matrix = self.ratings_df.pivot(
            index='userId', columns='movieId', values='rating').fillna(0)
        
        self.genre_matrix = self.movies_df[genre_columns].astype(float)
        self.genre_matrix.index = self.movies_df['movieId']

        self.user_means = self.ratings_df.groupby('userId')['rating'].mean()

        print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies")

    def compute_item_diversity(self, item1: int, item2: int) -> float:
        # Check cache first
        cache_key = tuple(sorted([item1, item2]))
        if cache_key in self.item_diversity_cache:
            return self.item_diversity_cache[cache_key]

        if item1 not in self.genre_matrix.index or item2 not in self.genre_matrix.index:
            return 0.0
        
        genres1 = self.genre_matrix.loc[item1]
        genres2 = self.genre_matrix.loc[item2]
        
        diversity = cosine(genres1, genres2)
        self.item_diversity_cache[cache_key] = diversity
        return diversity

    def get_user_similarity(self, user1: int, user2: int) -> float:
        # Check cache first
        cache_key = tuple(sorted([user1, user2]))
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]

        user1_ratings = self.user_movie_matrix.loc[user1]
        user2_ratings = self.user_movie_matrix.loc[user2]
        
        common_items = (user1_ratings > 0) & (user2_ratings > 0)
        if common_items.sum() < 5:
            return 0.0

        correlation = np.corrcoef(user1_ratings[common_items], 
                               user2_ratings[common_items])[0,1]
        
        similarity = max(correlation, 0) if not np.isnan(correlation) else 0.0
        self.user_similarity_cache[cache_key] = similarity
        return similarity

    def predict_user_rating(self, user_id: int, movie_id: int, k: int = 5) -> float:
        if movie_id not in self.user_movie_matrix.columns:
            return 0.0

        movie_raters = self.user_movie_matrix[self.user_movie_matrix[movie_id] > 0].index
        similarities = [(other_user, self.get_user_similarity(user_id, other_user))
                       for other_user in movie_raters if other_user != user_id]
        
        similarities = sorted([(u, s) for u, s in similarities if s > 0], 
                            key=lambda x: x[1], reverse=True)[:k]

        if not similarities:
            return 0.0

        numerator = sum(sim * (self.user_movie_matrix.loc[other_user, movie_id] - 
                             self.user_means[other_user])
                       for other_user, sim in similarities)
        denominator = sum(sim for _, sim in similarities)

        if denominator == 0:
            return 0.0

        predicted_rating = self.user_means[user_id] + (numerator / denominator)
        return max(1.0, min(5.0, predicted_rating))

    def get_diverse_recommendations(self, 
                                  group: List[int], 
                                  n_items: int = 5, 
                                  diversity_weight: float = 0.3) -> List[Tuple[int, str, float]]:
        # Get unrated items efficiently
        rated_items = set()
        for user in group:
            user_rated = set(self.ratings_df[self.ratings_df['userId'] == user]['movieId'])
            rated_items.update(user_rated)
            
        # Get popular items only (items with at least 5 ratings)
        item_counts = self.ratings_df['movieId'].value_counts()
        popular_items = set(item_counts[item_counts >= 5].index)
        candidate_items = list(popular_items - rated_items)

        recommendations = []
        
        # Pre-calculate predictions for all users and items
        predictions = {}
        for item in candidate_items:
            item_predictions = [self.predict_user_rating(user, item) for user in group]
            if any(p > 0 for p in item_predictions):
                predictions[item] = np.mean([p for p in item_predictions if p > 0])

        while len(recommendations) < n_items and predictions:
            best_score = float('-inf')
            best_item = None
            
            for item in list(predictions.keys()):  # Convert to list to avoid runtime modification issues
                relevance = predictions[item]
                
                if recommendations:
                    diversity = np.mean([self.compute_item_diversity(item, rec[0]) 
                                       for rec in recommendations])
                else:
                    diversity = 1.0
                
                score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if score > best_score:
                    best_score = score
                    best_item = item
            
            if best_item is None:
                break
                
            title = self.movies_df[self.movies_df['movieId'] == best_item]['title'].iloc[0]
            recommendations.append((best_item, title, best_score))
            del predictions[best_item]
        
        return recommendations

    def evaluate_diversity(self, recommendations: List[Tuple[int, str, float]]) -> Dict[str, float]:
        if not recommendations:
            return {'intra_list_diversity': 0.0, 'genre_coverage': 0.0}
        
        items = [rec[0] for rec in recommendations]
        
        # Calculate intra-list diversity using cached values
        diversities = [self.compute_item_diversity(items[i], items[j])
                      for i in range(len(items))
                      for j in range(i + 1, len(items))]
        
        ild = np.mean(diversities) if diversities else 0.0
        
        # Calculate genre coverage efficiently
        genre_cols = self.movies_df.columns[5:24]
        recommended_genres = set()
        for item in items:
            genres = self.genre_matrix.loc[item]
            item_genres = set(genre_cols[genres == 1])
            recommended_genres.update(item_genres)
        
        genre_coverage = len(recommended_genres) / len(genre_cols)
        
        return {
            'intra_list_diversity': ild,
            'genre_coverage': genre_coverage
        }

def main():
    try:
        recommender = DiverseGroupRecommender()
        recommender.load_data()
        
        test_group = [1, 4, 7]
        print(f"\nGenerating recommendations for group: {test_group}")
        
        for weight in [0.0, 0.3, 0.6]:
            print(f"\nDiversity weight: {weight}")
            print("-" * 70)
            recommendations = recommender.get_diverse_recommendations(
                group=test_group,
                n_items=5,
                diversity_weight=weight
            )
            
            for movie_id, title, score in recommendations:
                print(f"Movie: {title:<50} Score: {score:.2f}")
                
            metrics = recommender.evaluate_diversity(recommendations)
            print("\nDiversity Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.3f}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()