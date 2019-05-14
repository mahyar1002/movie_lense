import ipdb
from helper import log
from helper import load_df, load_data
from get_data import tags_to_panda, movies_to_panda, rating_to_panda, profile_to_panda, rating_test, merge_movies_tags, \
    select_test_rating
import os
from content_embeding import Tfidf
from similarity import SimilarityPredictions
from collaborative_embeding import CollaborativeFiltering
import pandas as pd
from profile_embeding import ProfileFiltering
from helper import save_data
import math
from evaluation import plot_item_value_count, RMSE, MAE
import random
from sklearn.model_selection import train_test_split


class Main:

    def __init__(self):
        self.tags = None
        self.movies = None
        self.sim_model = None
        self.ratings = None
        self.test = None
        self.train = None
        self.movie_ids = []
        self.sim_items_content = None
        self.sim_items_collaborative = None
        self.sim_items_ensemble = None
        self.sim_user_profile = None
        self.sim_user_collaborative = None
        self.sim_user_ensemble = None
        self.profiles = None

    def get_dataframes(self):
        self.tags = tags_to_panda(os.path.join(cwd, "data", "ml-1m", "tags.csv"))
        self.movies = movies_to_panda(os.path.join(cwd, "data", "ml-1m", "movies.csv"))
        self.movie_ids = self.movies['movieId'].to_list()
        self.ratings = rating_to_panda(os.path.join(cwd, "data", "ml-1m", "ratings.csv"))
        self.profiles = profile_to_panda(os.path.join(cwd, "data", "ml-1m", "users.csv"))
        self.train, self.test = rating_test(os.path.join(cwd, "data", "ml-1m", "ratings.csv"), 0.1)

    def content_emd(self, method='svd'):
        merged = merge_movies_tags(self.movies, self.tags)
        tfidf = Tfidf(merged)
        ## methos can be svd or ae
        tfidf.apply(method=method)

    def content_sim(self, method='svd'):
        content_embds = load_df('content_embeddings_{}.pkl'.format(method))
        # log("Info", "Content Embeddings Shape: {}".format(content_embds.shape))
        # log('Info', '\n{}'.format(content_embds.head(5)))
        # log("Info", "###### Start predicting of content similarity ######")
        self.sim_model = SimilarityPredictions(content_embds, similarity_metric="cosine")
        # log("Info", "###### End predicting of content similarity ######")

    def get_movie_ids(self, indices):
        return [self.movie_ids[index] for index in indices]

    def show_similar_movies_content(self, item=1, top=20, method='svd'):
        index = self.movie_ids.index(item)
        similar_item = self.sim_model.predict_similar_items(seed_item=index, n=top, method=method)
        # log("Info", "###### Similarity \n{} ".format(similar_item.head(5)))
        items = self.get_movie_ids(similar_item['item_id'].values)
        # log("Info", "###### Similar Items {} ".format(items))
        # log("Info",
        #     "###### Similar Movies to {} are {}".format(item, self.movies.loc[self.movies['movieId'].isin(items)]))
        return similar_item

    def collaborative_emb(self, method='svd', type='item'):
        cf = CollaborativeFiltering(self.ratings, type=type)
        cf.apply(method=method)

    def apply_item_emds(self):
        for method in methods:
            self.collaborative_emb(method)
            self.content_emd(method)

    def collaborative_sim(self, method):
        collaborative_embds = load_df('collaborative_embeddings_{}.pkl'.format(method))
        # log("Info", "Collaborative Embeddings Shape: {}".format(collaborative_embds.shape))
        # log('Info', '\n{}'.format(collaborative_embds.head(5)))
        # log("Info", "###### Start predicting of collaborative similarity ######")
        self.sim_model = SimilarityPredictions(collaborative_embds, similarity_metric="cosine")
        # log("Info", "###### End predicting of collaborative similarity ######")

    def show_similar_movies_collaboration(self, item=1, top=20, method='svd'):
        similar_item = self.sim_model.predict_similar_items(seed_item=item, n=top, method=method)
        # log("Info", "###### Similarity \n{} ".format(similar_item.head(5)))
        items = similar_item['item_id'].values
        # log("Info",
        #     "###### Similar Movies to {} are {}".format(item, self.movies.loc[self.movies['movieId'].isin(items)]))
        return similar_item

    def ensemble_similarities(self, sim1, sim2, on='item_id', type=None):
        if type and type in ['content', 'collaborative']:
            merged = pd.merge(sim1, sim2, on=on, how='outer').fillna(0.5)
            col_name = 'avg_{}'.format(type)
            merged[col_name] = (merged['similarity_score_svd'] + merged['similarity_score_ae']) / 2
            return merged.sort_values(col_name, ascending=False)
        else:
            merged = pd.merge(sim1, sim2, on=on, how='outer', suffixes=('', '_y')).fillna(0.5)
            merged.drop(list(merged.filter(regex='_y$')), axis=1, inplace=True)
            merged['avg'] = (merged['avg_content'] + merged['avg_collaborative']) / 2
            return merged.sort_values('avg', ascending=False)

    def collaborative_similarity(self, item):
        ## Collaborative Similarity [on movies]
        for method in methods:
            self.collaborative_sim(method)
            similar_items.append(self.show_similar_movies_collaboration(item=item, top=1000, method=method))
        sim_items_collaborative = self.ensemble_similarities(similar_items[0], similar_items[1], type='collaborative')
        sim_items_collaborative['movieId'] = sim_items_collaborative['item_id'].values
        self.sim_items_collaborative = pd.merge(sim_items_collaborative, self.movies, on="movieId", how='inner')

    def content_similarity(self, item):
        ## Content Similarity [on movies]
        similar_items.clear()
        for method in methods:
            self.content_sim(method)
            similar_items.append(self.show_similar_movies_content(item=item, top=1000, method=method))
        sim_items_content = self.ensemble_similarities(similar_items[0], similar_items[1], type='content')
        sim_items_content['movieId'] = self.get_movie_ids(sim_items_content['item_id'].values)
        self.sim_items_content = pd.merge(sim_items_content, self.movies, on="movieId", how='inner')

    def ensemble_item_similarity(self):
        ## Ensemble Similarities [content & movie]
        self.sim_items_ensemble = self.ensemble_similarities(self.sim_items_collaborative, self.sim_items_content,
                                                             on='movieId')
        self.sim_items_ensemble.reset_index(inplace=True)
        self.sim_items_ensemble.drop(0, inplace=True)
        # log("Info",
        #     "###### Ensemble Similar Items [Collaborative Base] \n{}".format(self.sim_items_collaborative.head(10)))
        # log("Info", "###### Ensemble Similar Items [Content Base] \n{}".format(self.sim_items_content.head(10)))
        # log("Info", "###### Ensemble Similar Items \n{}".format(self.sim_items_ensemble.head(10)))

    def apply_item_similarity(self, item):
        self.collaborative_similarity(item)
        self.content_similarity(item)
        self.ensemble_item_similarity()

    def profile_emd(self):
        pf = ProfileFiltering(self.profiles)
        pf.pca_to_profile()

    def profile_similarity(self, user):
        profile_embds = load_df('profile_embeddings_pca.pkl')
        # log("Info", "Profile Embeddings Shape: {}".format(profile_embds.shape))
        # log('Info', '\n{}'.format(profile_embds.head(5)))
        # log("Info", "###### Start predicting of profile similarity ######")
        self.sim_model = SimilarityPredictions(profile_embds, similarity_metric="cosine")
        self.sim_user_profile = self.show_similar_users_profile(user=user, top=100)
        self.sim_user_profile['similarity_score_pca'] = self.sim_user_profile['similarity_score_pca'] - 0.4
        # log("Info", "###### End predicting of profile similarity ######")

    def show_similar_users_profile(self, user=1, top=20):
        index = user - 1
        similar_users = self.sim_model.predict_similar_items(seed_item=index, n=top, method='pca')
        similar_users['item_id'] = similar_users['item_id'].values + 1
        # log("Info", "###### Similarity Between Users \n{} ".format(similar_users.head(5)))
        return similar_users

    def user_collaborative_emd(self):
        cf = CollaborativeFiltering(self.ratings, type='user')
        cf.apply(method='svd')

    def user_collaborative_similarity(self, user):
        collaborative_embds = load_df('collaborative_embeddings_user_svd.pkl')
        # log("Info", "Collaborative Embeddings Shape: {}".format(collaborative_embds.shape))
        # log('Info', '\n{}'.format(collaborative_embds.head(5)))
        # log("Info", "###### Start predicting of collaborative similarity ######")
        self.sim_model = SimilarityPredictions(collaborative_embds, similarity_metric="cosine")
        self.sim_user_collaborative = self.show_similar_user_collaborative(user=user, top=100)
        # log("Info", "###### End predicting of collaborative similarity ######")

    def show_similar_user_collaborative(self, user=1, top=20):
        similar_users = self.sim_model.predict_similar_items(seed_item=user, n=top, method='svd')
        # log("Info", "###### Similarity Between Users \n{} ".format(similar_users.head(5)))
        return similar_users

    def ensemble_user_similarity(self):
        merged = pd.merge(self.sim_user_profile, self.sim_user_collaborative, on='item_id', how='outer',
                          suffixes=('', '_y')).fillna(0.5)
        merged.drop(list(merged.filter(regex='_y$')), axis=1, inplace=True)
        merged['avg'] = (merged['similarity_score_svd'] + (merged['similarity_score_pca'])) / 2
        self.sim_user_ensemble = merged.sort_values('avg', ascending=False)
        self.sim_user_ensemble.rename(columns={'item_id': 'userId'}, inplace=True)
        self.sim_user_ensemble.reset_index(inplace=True)
        self.sim_user_ensemble.drop(0, inplace=True)

    def apply_user_similarity(self, user):
        self.profile_similarity(user)
        self.user_collaborative_similarity(user)
        self.ensemble_user_similarity()

    def apply_user_emds(self):
        self.profile_emd()
        self.user_collaborative_emd()

    def get_avg_movie_rate(self, user, movies):
        users = self.ratings.loc[self.ratings['userId'] == user]
        filtered = users.loc[users['movieId'].isin(movies)]
        return filtered['rating'].mean() if filtered['rating'].mean() < 5 else 5

    def get_avg_user_rate(self, movie, users):
        users = self.ratings.loc[self.ratings['userId'].isin(users)]
        filtered = users.loc[users['movieId'] == movie]
        return filtered['rating'].mean() if filtered['rating'].mean() < 5 else 5

    def calc_score(self, item, user):
        self.apply_item_similarity(item)
        self.apply_user_similarity(user)

        top_five_items = self.sim_items_ensemble.head(10)['movieId'].values
        top_five_users = self.sim_user_ensemble.head(10)['userId'].values

        avg_movie_rate = self.get_avg_movie_rate(user, top_five_items)
        avg_user_rate = self.get_avg_user_rate(item, top_five_users)

        if math.isnan(avg_movie_rate):
            return avg_user_rate

        if math.isnan(avg_user_rate):
            return avg_movie_rate

        return (avg_movie_rate + avg_user_rate) / 2

    #### other methods #####
    #### item similarity ####
    def get_mean_rating(self, user):
        user_ratings = self.ratings.loc[self.ratings['userId'] == user]
        return user_ratings['rating'].mean() if user_ratings['rating'].mean() < 5 else 5

    def calc_score_item_base(self, item, user):
        self.apply_item_similarity(item)

        top_five_items = self.sim_items_ensemble.head(10)['movieId'].values

        avg_movie_rate = self.get_avg_movie_rate(user, top_five_items)

        if math.isnan(avg_movie_rate):
            return self.get_mean_rating(user)

        return avg_movie_rate

    #### user similarity ####
    def calc_score_user_base(self, item, user):
        self.apply_user_similarity(user)

        top_five_users = self.sim_user_ensemble.head(10)['userId'].values

        avg_user_rate = self.get_avg_user_rate(item, top_five_users)

        if math.isnan(avg_user_rate):
            return self.get_mean_rating(user)

        return avg_user_rate
    ####

    def recommender(self):
        for p in range(2, 11, 2):
            selected_test_rating = select_test_rating(self.ratings, p)
            x = round(500 / selected_test_rating.shape[0], 4)
            train, test = train_test_split(selected_test_rating, test_size=x)

            result = []
            def_perc = 0
            counter = 0
            for index, row in test.iterrows():
                predict = self.calc_score_user_base(row['movieId'], row['userId'])
                result.append((round(predict, 1), row['rating']))
                counter += 1
                print("number of satisfied items for p={}: {} --- {}%".format(p, counter, def_perc))
                perc = math.floor((counter * 100) / test.shape[0])
                if perc > def_perc:
                    def_perc = perc

            log("Info", "result for p={}:\n{}".format(p, result))
            save_data(result, "result/predict_test_data_{}.pkl".format(p))

    def evaluate(self, path, p):
        results = load_data(path)
        rmse = RMSE(results)
        mae = MAE(results)
        log("Info", "p: {}, RMSE: {}, MAE: {}".format(p, rmse, mae))


if __name__ == "__main__":
    cwd = os.getcwd()
    methods = ['svd', 'ae']
    similar_items = []
    app = Main()
    app.get_dataframes()

    # plot_item_value_count(app.ratings)

    # app.recommender()

    for item in range(2, 11, 2):
        app.evaluate('result/back/rmsa/2/predict_test_data_{}.pkl'.format(item), item)

    # app.apply_item_emds()
    # app.apply_item_similarity()

    # app.apply_user_emds()
    # app.apply_user_similarity()