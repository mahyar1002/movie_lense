import ipdb
from helper import log
from helper import load_df
from get_data import merge_movies_tags, movies_to_panda, tags_to_panda, rating_to_panda, profile_to_panda
import os
from content_embeding import Tfidf
from similarity import SimilarityPredictions
from collaborative_embeding import CollaborativeFiltering
import pandas as pd
from profile_embeding import ProfileFiltering


class Main:

    def __init__(self):
        self.tags = None
        self.movies = None
        self.sim_model = None
        self.ratings = None
        self.movie_ids = []
        self.sim_items_content = None
        self.sim_items_collaborative = None
        self.sim_items_ensemble = None
        self.profiles = None

    def get_dataframes(self):
        self.tags = tags_to_panda(os.path.join(cwd, "data", "ml-1m", "tags.csv"))
        self.movies = movies_to_panda(os.path.join(cwd, "data", "ml-1m", "movies.csv"))
        self.movie_ids = self.movies['movieId'].to_list()
        self.ratings = rating_to_panda(os.path.join(cwd, "data", "ml-1m", "ratings.csv"))
        self.profiles = profile_to_panda(os.path.join(cwd, "data", "ml-1m", "users.csv"))

    def content_emd(self, method='svd'):
        merged = merge_movies_tags(self.movies, self.tags)
        tfidf = Tfidf(merged)
        ## methos can be svd or ae
        tfidf.apply(method=method)

    def content_sim(self, method='svd'):
        content_embds = load_df('content_embeddings_{}.pkl'.format(method))
        log("Info", "Content Embeddings Shape: {}".format(content_embds.shape))
        log('Info', '\n{}'.format(content_embds.head(5)))
        log("Info", "###### Start predicting of content similarity ######")
        self.sim_model = SimilarityPredictions(content_embds, similarity_metric="cosine")
        log("Info", "###### End predicting of content similarity ######")

    def get_movie_ids(self, indices):
        return [self.movie_ids[index] for index in indices]

    def show_similar_movies_content(self, item=1, top=20, method='svd'):
        index = self.movie_ids.index(item)
        similar_item = self.sim_model.predict_similar_items(seed_item=index, n=top, method=method)
        log("Info", "###### Similarity \n{} ".format(similar_item.head(5)))
        items = self.get_movie_ids(similar_item['item_id'].values)
        # log("Info", "###### Similar Items {} ".format(items))
        log("Info",
            "###### Similar Movies to {} are {}".format(item, self.movies.loc[self.movies['movieId'].isin(items)]))
        return similar_item

    def collaborative_emb(self, method='svd'):
        cf = CollaborativeFiltering(self.ratings)
        cf.apply(method=method)

    def apply_item_emds(self):
        for method in methods:
            self.collaborative_emb(method)
            self.content_emd(method)

    def collaborative_sim(self, method):
        collaborative_embds = load_df('collaborative_embeddings_{}.pkl'.format(method))
        log("Info", "Collaborative Embeddings Shape: {}".format(collaborative_embds.shape))
        log('Info', '\n{}'.format(collaborative_embds.head(5)))
        log("Info", "###### Start predicting of collaborative similarity ######")
        self.sim_model = SimilarityPredictions(collaborative_embds, similarity_metric="cosine")
        log("Info", "###### End predicting of collaborative similarity ######")

    def show_similar_movies_collaboration(self, item=1, top=20, method='svd'):
        similar_item = self.sim_model.predict_similar_items(seed_item=item, n=top, method=method)
        log("Info", "###### Similarity \n{} ".format(similar_item.head(5)))
        items = similar_item['item_id'].values
        # log("Info", "###### Similar Items {} ".format(items))
        log("Info",
            "###### Similar Movies to {} are {}".format(item, self.movies.loc[self.movies['movieId'].isin(items)]))
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

    def collaborative_similarity(self):
        ## Collaborative Similarity [on movies]
        # for method in methods:
        #     app.collaborative_emb(method)
        for method in methods:
            self.collaborative_sim(method)
            similar_items.append(self.show_similar_movies_collaboration(item=1, top=1000, method=method))
        sim_items_collaborative = self.ensemble_similarities(similar_items[0], similar_items[1], type='collaborative')
        sim_items_collaborative['movieId'] = sim_items_collaborative['item_id'].values
        self.sim_items_collaborative = pd.merge(sim_items_collaborative, self.movies, on="movieId", how='inner')

    def content_similarity(self):
        ## Content Similarity [on movies]
        # for method in methods:
        #     app.content_emd(method)
        similar_items.clear()
        for method in methods:
            self.content_sim(method)
            similar_items.append(self.show_similar_movies_content(item=1, top=1000, method=method))
        sim_items_content = self.ensemble_similarities(similar_items[0], similar_items[1], type='content')
        sim_items_content['movieId'] = self.get_movie_ids(sim_items_content['item_id'].values)
        self.sim_items_content = pd.merge(sim_items_content, self.movies, on="movieId", how='inner')

    def ensemble_item_similarity(self):
        ## Ensemble Similarities [content & movie]
        result = self.ensemble_similarities(self.sim_items_collaborative, self.sim_items_content, on='movieId')

        log("Info",
            "###### Ensemble Similar Items [Collaborative Base] \n{}".format(self.sim_items_collaborative.head(10)))
        log("Info", "###### Ensemble Similar Items [Content Base] \n{}".format(self.sim_items_content.head(10)))
        log("Info", "###### Ensemble Similar Items \n{}".format(result.head(10)))

    def apply_item_similarity(self):
        self.collaborative_similarity()
        self.content_similarity()
        self.ensemble_item_similarity()

    def profile_emd(self):
        pf = ProfileFiltering(self.profiles)
        pf.pca_to_profile()


if __name__ == "__main__":
    cwd = os.getcwd()
    methods = ['svd', 'ae']
    similar_items = []
    app = Main()
    app.get_dataframes()
    app.collaborative_emb('ae')
    # app.apply_item_emds()
    # app.apply_item_similarity()

    # app.profile_emd()
