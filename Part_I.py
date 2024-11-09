import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class MovieLens100kRecommender:

    def __init__(self):
        """Initialize the MovieLens 100K recommender system"""
        self.ratings_df = None
        self.movies_df = None
        self.user_movie_matrix = None
        self.user_means = None

    def load_data(self,
                  data_path: str = 'u.data',
                  item_path: str = 'u.item') -> None:
        """Load the MovieLens 100K dataset"""
        # Set pandas display options for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)

        # Load ratings
        self.ratings_df = pd.read_csv(
            data_path,
            sep='\t',
            names=['userId', 'movieId', 'rating', 'timestamp'])

        # Convert timestamp to datetime and format it
        self.ratings_df['timestamp'] = pd.to_datetime(
            self.ratings_df['timestamp'], unit='s')
        self.ratings_df['timestamp'] = self.ratings_df[
            'timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Load movies
        columns = [
            'movieId', 'title', 'release_date', 'video_release_date',
            'imdb_url'
        ]
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir',
            'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
            'War', 'Western'
        ]
        columns.extend(genre_columns)

        self.movies_df = pd.read_csv(item_path,
                                     sep='|',
                                     names=columns,
                                     encoding='latin-1')

        # Merge ratings with movie titles
        ratings_with_titles = pd.merge(self.ratings_df,
                                       self.movies_df[['movieId', 'title']],
                                       on='movieId')

        # Reorder columns and format them
        ratings_with_titles = ratings_with_titles[[
            'userId', 'movieId', 'title', 'rating', 'timestamp'
        ]]

        # Create user-movie matrix
        self.user_movie_matrix = self.ratings_df.pivot(
            index='userId', columns='movieId', values='rating').fillna(0)

        # Calculate user means
        self.user_means = self.ratings_df.groupby('userId')['rating'].mean()

        print("Dataset Summary:")
        print(f"Number of ratings: {len(self.ratings_df)}")
        print(f"Number of users: {len(self.ratings_df['userId'].unique())}")
        print(f"Number of movies: {len(self.ratings_df['movieId'].unique())}")

        print(
            "\nFirst few ratings with movie titles and formatted timestamps:")
        print("-" * 100)
        # Format the display using a formatted string for each row
        print(
            f"{'userId':>7}  {'movieId':>7}  {'title':<40}  {'rating':>6}  {'timestamp':<19}"
        )
        print("-" * 100)
        for _, row in ratings_with_titles.head().iterrows():
            print(
                f"{row['userId']:>7}  {row['movieId']:>7}  {row['title']:<40}  {row['rating']:>6}  {row['timestamp']}"
            )

        # print("\nExample Movies in Dataset:")
        # print("-" * 60)
        # print(f"{'movieId':>7}  {'title':<50}")
        # print("-" * 60)
        # for _, row in self.movies_df.head().iterrows():
        #     print(f"{row['movieId']:>7}  {row['title']:<50}")

    def pearson_similarity(self, user1: int, user2: int) -> float:
        """Calculate Pearson correlation between two users"""
        user1_ratings = self.user_movie_matrix.loc[user1]
        user2_ratings = self.user_movie_matrix.loc[user2]

        # Find common rated movies
        common_movies = (user1_ratings != 0) & (user2_ratings != 0)

        if common_movies.sum() < 2:
            return 0

        user1_common = user1_ratings[common_movies]
        user2_common = user2_ratings[common_movies]

        if len(set(user1_common)) == 1 or len(set(user2_common)) == 1:
            return 0

        try:
            correlation, _ = pearsonr(user1_common, user2_common)
            return correlation if not np.isnan(correlation) else 0
        except:
            return 0

    def custom_similarity(self, user1: int, user2: int) -> float:
        """Custom similarity function"""
        # Basic Pearson correlation
        pearson_sim = self.pearson_similarity(user1, user2)

        # Genre preference similarity
        user1_movies = self.ratings_df[self.ratings_df['userId'] ==
                                       user1]['movieId']
        user2_movies = self.ratings_df[self.ratings_df['userId'] ==
                                       user2]['movieId']

        genre_cols = self.movies_df.columns[5:24]
        user1_genres = self.movies_df[self.movies_df['movieId'].isin(
            user1_movies)][genre_cols].mean()
        user2_genres = self.movies_df[self.movies_df['movieId'].isin(
            user2_movies)][genre_cols].mean()

        genre_sim = 1 - np.mean(np.abs(user1_genres - user2_genres))

        # Rating pattern similarity
        rating_counts1 = self.ratings_df[self.ratings_df['userId'] == user1][
            'rating'].value_counts().sort_index()
        rating_counts2 = self.ratings_df[self.ratings_df['userId'] == user2][
            'rating'].value_counts().sort_index()

        all_ratings = sorted(
            set(rating_counts1.index) | set(rating_counts2.index))
        pattern1 = np.array([rating_counts1.get(r, 0) for r in all_ratings])
        pattern2 = np.array([rating_counts2.get(r, 0) for r in all_ratings])

        pattern_sim = 1 - np.sum(
            np.abs(pattern1 / np.sum(pattern1) - pattern2 / np.sum(pattern2)))

        return 0.4 * pearson_sim + 0.3 * genre_sim + 0.3 * pattern_sim

    def predict_rating(self,
                       user_id: int,
                       movie_id: int,
                       k: int = 5,
                       use_custom_similarity: bool = False) -> float:
        """Predict rating for a user-movie pair"""
        if movie_id not in self.user_movie_matrix.columns:
            return 0

        movie_raters = self.user_movie_matrix[self.user_movie_matrix[movie_id]
                                              > 0].index

        similarities = []
        for other_user in movie_raters:
            if other_user != user_id:
                sim = self.custom_similarity(user_id, other_user) if use_custom_similarity \
                    else self.pearson_similarity(user_id, other_user)
                if sim > 0:
                    similarities.append((other_user, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_users = similarities[:k]

        if not top_k_users:
            movie_ratings = self.user_movie_matrix[movie_id][
                self.user_movie_matrix[movie_id] > 0]
            return float(movie_ratings.mean()) if len(movie_ratings) > 0 else 0

        numerator = sum(sim *
                        (self.user_movie_matrix.loc[other_user, movie_id] -
                         self.user_means[other_user])
                        for other_user, sim in top_k_users)
        denominator = sum(abs(sim) for _, sim in top_k_users)

        if denominator == 0:
            return 0

        predicted_rating = self.user_means[user_id] + (numerator / denominator)

        # Clamp the predicted rating between 1 and 5
        return max(1.0, min(5.0, predicted_rating))

    def get_recommendations(self,
                            user_id: int,
                            n: int = 10,
                            use_custom_similarity: bool = False
                            ) -> List[Tuple[int, str, float]]:
        """Get movie recommendations for a user"""
        unrated_movies = self.user_movie_matrix.columns[
            self.user_movie_matrix.loc[user_id] == 0]
        popular_movies = [
            movie for movie in unrated_movies
            if (self.user_movie_matrix[movie] > 0).sum() >= 5
        ]

        predictions = []
        for movie_id in popular_movies:
            pred_rating = self.predict_rating(
                user_id, movie_id, use_custom_similarity=use_custom_similarity)
            if pred_rating > 0:
                movie_title = self.movies_df[self.movies_df['movieId'] ==
                                             movie_id]['title'].iloc[0]
                predictions.append((movie_id, movie_title, pred_rating))

        return sorted(predictions, key=lambda x: x[2], reverse=True)[:n]

    def get_group_recommendations(
            self,
            group_users: List[int],
            n: int = 10,
            method: str = 'disagreement_aware'
    ) -> List[Tuple[int, str, float]]:
        """Get group recommendations"""
        all_predictions = defaultdict(list)

        for user_id in group_users:
            for movie_id in self.user_movie_matrix.columns:
                if self.user_movie_matrix.loc[user_id, movie_id] == 0:
                    pred = self.predict_rating(user_id, movie_id)
                    if pred > 0:
                        all_predictions[movie_id].append(pred)

        final_scores = []
        for movie_id, predictions in all_predictions.items():
            if len(predictions) == len(group_users):
                if method == 'average':
                    score = np.mean(predictions)
                elif method == 'least_misery':
                    score = min(predictions)
                else:
                    disagreement = np.std(predictions)
                    score = np.mean(predictions) * (1 - 0.3 * disagreement)

                movie_title = self.movies_df[self.movies_df['movieId'] ==
                                             movie_id]['title'].iloc[0]
                final_scores.append((movie_id, movie_title, score))

        return sorted(final_scores, key=lambda x: x[2], reverse=True)[:n]

    def analyze_group_disagreements(self,
                                    group_users: List[int],
                                    n_movies: int = 10) -> None:
        """Visualize group disagreements on movies"""
        # Clear any existing plots
        plt.clf()

        movies = self.movies_df['movieId'].head(n_movies)
        disagreements = []
        titles = []

        for movie_id in movies:
            predictions = []
            for user_id in group_users:
                pred = self.predict_rating(user_id, movie_id)
                if pred > 0:
                    predictions.append(pred)

            if predictions:
                disagreements.append(np.std(predictions))
                titles.append(self.movies_df[self.movies_df['movieId'] ==
                                             movie_id]['title'].iloc[0])

        plt.figure(figsize=(15, 6))
        plt.bar(range(len(disagreements)), disagreements)
        plt.xticks(range(len(disagreements)), titles, rotation=45, ha='right')
        plt.title('Group Disagreements on Movies')
        plt.xlabel('Movie')
        plt.ylabel('Disagreement Score (Standard Deviation)')
        plt.tight_layout()

        # Don't call plt.show() here, it will be called in main


if __name__ == "__main__":
    # Initialize recommender system
    recommender = MovieLens100kRecommender()

    # Load data
    recommender.load_data()

    # Get recommendations for user 1
    print("\nTop Movie Recommendations for User 1 (using Pearson similarity):")
    print("-" * 70)
    recommendations = recommender.get_recommendations(
        user_id=1, n=5, use_custom_similarity=False)
    for movie_id, title, predicted_rating in recommendations:
        print(f"Movie: {title:<50} Rating: {predicted_rating:.2f}/5.00")

    print("\nTop Movie Recommendations for User 1 (using custom similarity):")
    print("-" * 70)
    recommendations = recommender.get_recommendations(
        user_id=1, n=5, use_custom_similarity=True)
    for movie_id, title, predicted_rating in recommendations:
        print(f"Movie: {title:<50} Rating: {predicted_rating:.2f}/5.00")

    # Get group recommendations
    group_users = [1, 2, 3, 4]
    print(f"\nGroup Movie Recommendations for users {group_users}:")
    print("-" * 70)
    group_recommendations = recommender.get_group_recommendations(
        group_users=group_users, n=5, method='disagreement_aware')
    for movie_id, title, score in group_recommendations:
        print(f"Movie: {title:<50} Score: {score:.2f}/5.00")

    # Analyze and display group disagreements
    print("\nAnalyzing group disagreements...")
    recommender.analyze_group_disagreements(group_users)
    plt.show()  # Explicitly call plt.show()
