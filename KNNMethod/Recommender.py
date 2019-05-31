import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import os
import gc


class KNNRecommender:

    def __init__(self):
        self.model = NearestNeighbors()

    def set_model_parameters(self, n_neighbors, algorithm, metric, n_jobs=None):

        """
         The algorithms are: {'auto', 'ball_tree', 'kd_tree', 'brute'}
         The Matrices are : {'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'}
        """
        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def get_data(self):

        movies_df = pd.read_csv(
            "Movies Data/movies.csv",
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
        ratings_df = pd.read_csv(
            "Movies Data/ratings.csv",
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        df_movies_count = pd.DataFrame(
            ratings_df.groupby('movieId').size(),
            columns=['count'])

        popular_movies = list(set(df_movies_count.query('count >= 10').index))
        movies_filter = ratings_df.movieId.isin(popular_movies).values

        df_users_count = pd.DataFrame(
            ratings_df.groupby('userId').size(),
            columns=['count'])

        active_users = list(set(df_users_count.query('count >= 10').index))
        users_filter = ratings_df.userId.isin(active_users).values

        df_ratings_filtered = ratings_df[movies_filter & users_filter]

        movies_users_matrix = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)

        movies_dict = {
            movie: i for i, movie in
            enumerate(list(movies_df.set_index('movieId').loc[movies_users_matrix.index].title))
        }
        movie_user_mat_sparse = csr_matrix(movies_users_matrix.values)

        del movies_df, df_movies_count, df_users_count
        del ratings_df, df_ratings_filtered, movies_users_matrix
        gc.collect()
        return movie_user_mat_sparse, movies_dict

    def get_similar_using_fuzzy_matching(self, movies_dict, movie_name):

        matching_tuple = []
        # get match
        for title, idx in movies_dict.items():
            ratio = fuzz.ratio(title.lower(), movie_name.lower())
            if ratio >= 60:
                matching_tuple.append((title, idx, ratio))

        matching_tuple = sorted(matching_tuple, key=lambda x: x[2])[::-1]
        if matching_tuple:
            return matching_tuple[0][1]

    def model_fit(self, model, data, movies_dict, movie_name, n_recommendations):

        model.fit(data)
        idx = self.get_similar_using_fuzzy_matching(movies_dict, movie_name)
        if idx is None:
            return -1
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)

        'get indices of recommendations'
        recommendations = \
            sorted(
                list(
                    zip(indices.squeeze().tolist(), distances.squeeze().tolist())
                   ),
                key=lambda x: x[1]
            )[:0:-1]

        'return recommendation as list of tuples like (movieId, distance)'
        return recommendations

    def get_similar_movie(self, movie_name, recommendations_number=10):

        movie_user_mat_sparse, movies_dict = self.get_data()

        recommendations = self.model_fit(self.model, movie_user_mat_sparse, movies_dict, movie_name, recommendations_number)
        if recommendations == -1:
            print("Error in model fitting")
            return -1
        else:
            result = []
            reverse_movies_dict = {value: key for key, value in movies_dict.items()}
            for i, (idx, dist) in enumerate(recommendations):
                result.append((reverse_movies_dict[idx], dist))
            result.sort(key= lambda tup: tup[1])
            return result


def main(movie_name, recommendations_number):
    recommender = KNNRecommender()
    recommender.set_model_parameters(20, 'auto', 'cosine', -1)
    result = recommender.get_similar_movie(movie_name, recommendations_number)
    if result == -1:
        return -1
    else:
        return result


if __name__ == '__main__':
    result = main("interstellar", 12)
    print(result)

