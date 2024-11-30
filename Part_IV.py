import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

class CounterfactualGroupRecommender:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.user_movie_matrix = None
        self.user_means = None
        self.similarity_cache = {}
        self.movie_titles = {}
        self.user_item_ratings = defaultdict(dict)  # Cache for user-item ratings

    def load_data(self, data_path: str = 'u.data', item_path: str = 'u.item') -> None:
        """Load MovieLens dataset"""
        # Load ratings
        self.ratings_df = pd.read_csv(
            data_path,
            sep='\t',
            names=['userId', 'movieId', 'rating', 'timestamp'])

        # Load movies
        columns = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url']
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        columns.extend(genre_columns)

        self.movies_df = pd.read_csv(item_path, sep='|', names=columns, encoding='latin-1')
        self.movie_titles = dict(zip(self.movies_df['movieId'], self.movies_df['title']))

        # Create user-movie matrix
        self.user_movie_matrix = self.ratings_df.pivot(
            index='userId', columns='movieId', values='rating').fillna(0)
        
        # Calculate user means
        self.user_means = self.ratings_df.groupby('userId')['rating'].mean()

        # Build user-item ratings cache
        for _, row in self.ratings_df.iterrows():
            self.user_item_ratings[row['userId']][row['movieId']] = row['rating']
        
        print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies")

    def get_user_similarity(self, user1: int, user2: int) -> float:
        """Compute similarity between users with caching"""
        cache_key = tuple(sorted([user1, user2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Get ratings for both users
        items1 = self.user_item_ratings[user1]
        items2 = self.user_item_ratings[user2]
        
        # Find common items
        common_items = set(items1.keys()) & set(items2.keys())
        if len(common_items) < 5:
            return 0.0

        # Calculate correlation
        ratings1 = [items1[item] for item in common_items]
        ratings2 = [items2[item] for item in common_items]
        
        correlation = np.corrcoef(ratings1, ratings2)[0,1]
        similarity = max(correlation, 0) if not np.isnan(correlation) else 0.0
        
        self.similarity_cache[cache_key] = similarity
        return similarity

    def predict_rating(self, user_id: int, movie_id: int, k: int = 5) -> float:
        """Predict rating for a user-movie pair"""
        if movie_id not in self.user_movie_matrix.columns:
            return 0.0

        # Get users who rated this movie
        rated_users = [u for u, items in self.user_item_ratings.items() 
                      if movie_id in items and u != user_id]
        
        if not rated_users:
            return 0.0

        # Get top k similar users
        similarities = [(u, self.get_user_similarity(user_id, u)) 
                       for u in rated_users]
        similarities = sorted([(u, s) for u, s in similarities if s > 0], 
                            key=lambda x: x[1], reverse=True)[:k]

        if not similarities:
            return 0.0

        # Calculate weighted average
        numerator = sum(sim * (self.user_item_ratings[u][movie_id] - self.user_means[u])
                       for u, sim in similarities)
        denominator = sum(sim for _, sim in similarities)

        if denominator == 0:
            return 0.0

        predicted_rating = self.user_means[user_id] + (numerator / denominator)
        return max(1.0, min(5.0, predicted_rating))

    def get_movies_to_explain(self, group: List[int], n: int = 5) -> List[int]:
        """Get movies that the group might be interested in"""
        # Get movies rated highly by at least one group member
        group_ratings = self.ratings_df[self.ratings_df['userId'].isin(group)]
        highly_rated = group_ratings[group_ratings['rating'] >= 4]['movieId'].unique()
        
        # Get popular movies among these
        movie_ratings = self.ratings_df[self.ratings_df['movieId'].isin(highly_rated)]
        popular_movies = movie_ratings['movieId'].value_counts()
        
        return popular_movies.head(n).index.tolist()

    def generate_counterfactual(self, group: List[int], target_item: int) -> Dict:
        """Generate counterfactual explanation"""
        # Get original group rating
        group_ratings = []
        for user in group:
            if target_item in self.user_item_ratings[user]:
                group_ratings.append(self.user_item_ratings[user][target_item])
            else:
                pred = self.predict_rating(user, target_item)
                if pred > 0:
                    group_ratings.append(pred)
        
        original_score = np.mean(group_ratings) if group_ratings else 0
        if original_score == 0:
            return {'explanation': f"No prediction available for {self.movie_titles[target_item]}"}

        # Find influential movies
        influential_items = []
        affected_users = set()
        max_impact = 0

        # Get movies rated by group members
        group_movies = set()
        for user in group:
            group_movies.update(self.user_item_ratings[user].keys())

        for movie in group_movies:
            if movie == target_item:
                continue

            # Count users who rated this movie
            users_rated = [u for u in group if movie in self.user_item_ratings[u]]
            if len(users_rated) < len(group) * 0.5:
                continue

            # Calculate impact of removing this movie
            temp_score = original_score
            temp_affected = set()
            
            for user in users_rated:
                original_rating = self.user_item_ratings[user][movie]
                del self.user_item_ratings[user][movie]
                
                new_pred = self.predict_rating(user, target_item)
                if abs(new_pred - original_score) > 0.5:  # Significant change threshold
                    temp_affected.add(user)
                    temp_score = new_pred
                
                # Restore the rating
                self.user_item_ratings[user][movie] = original_rating

            impact = abs(original_score - temp_score)
            if impact > max_impact and temp_affected:
                influential_items = [movie]
                affected_users = temp_affected
                max_impact = impact

        return {
            'removed_items': influential_items,
            'affected_users': list(affected_users),
            'impact_score': max_impact,
            'original_score': original_score,
            'new_score': original_score - max_impact
        }

    def format_explanation(self, explanation: Dict) -> str:
        """Format the counterfactual explanation"""
        if 'explanation' in explanation:
            return explanation['explanation']

        if not explanation.get('removed_items'):
            return "No clear explanation found for this recommendation."

        movie_names = [self.movie_titles[item] for item in explanation['removed_items']]
        movies_str = ", ".join(movie_names)

        return (
            f"This movie was recommended because the group watched: {movies_str}\n"
            f"Impact: {explanation['impact_score']:.2f} rating points\n"
            f"Number of affected users: {len(explanation['affected_users'])}\n"
            f"Original rating: {explanation['original_score']:.2f}\n"
            f"Rating without these movies: {explanation['new_score']:.2f}"
        )

def main():
    try:
        print("Initializing recommender system...")
        recommender = CounterfactualGroupRecommender()
        recommender.load_data()

        # Test group
        test_group = [1, 4, 7]
        print(f"\nGenerating explanations for group: {test_group}")

        # Get movies to explain
        movies_to_explain = recommender.get_movies_to_explain(test_group)
        
        for movie_id in movies_to_explain:
            print(f"\nGenerating explanation for: {recommender.movie_titles[movie_id]}")
            explanation = recommender.generate_counterfactual(test_group, movie_id)
            print("\nCounterfactual Explanation:")
            print(recommender.format_explanation(explanation))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()