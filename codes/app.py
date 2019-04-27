import ipdb
from helper import log
from helper import load_df
from get_data import merge_movies_tags, movies_to_panda, tags_to_panda, rating_to_panda
import os
from content_embeding import Tfidf
from similarity import SimilarityPredictions
from collaborative_embeding import CollaborativeFiltering


class Main:

    def __init__(self):
        self.tags = None
        self.movies = None
        self.sim_model = None
        self.ratings = None
        self.movie_ids = []

    def get_dataframes(self):
        self.tags = tags_to_panda(os.path.join(cwd, "data", "ml-1m", "tags.csv"))
        self.movies = movies_to_panda(os.path.join(cwd, "data", "ml-1m", "movies.csv"))
        self.movie_ids = self.movies['movieId'].to_list()
        self.ratings = rating_to_panda(os.path.join(cwd, "data", "ml-1m", "ratings.csv"))

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

    def show_similar_movies_content(self, item=1, top=20):
        index = self.movie_ids.index(item)
        similar_item = self.sim_model.predict_similar_items(seed_item=index, n=top)
        log("Info", "###### Similarity {} ".format(similar_item))
        items = self.get_movie_ids(similar_item['item_id'].values())
        log("Info", "###### Similar Items {} ".format(items))
        log("Info",
            "###### Similar Movies to {} are {}".format(item, self.movies.loc[self.movies['movieId'].isin(items)]))

    def collaborative_emb(self, method='svd'):
        cf = CollaborativeFiltering(self.ratings)
        cf.apply(method=method)

    def collaborative_sim(self, method):
        collaborative_embds = load_df('collaborative_embeddings_{}.pkl'.format(method))
        log("Info", "Collaborative Embeddings Shape: {}".format(collaborative_embds.shape))
        log('Info', '\n{}'.format(collaborative_embds.head(5)))
        log("Info", "###### Start predicting of collaborative similarity ######")
        self.sim_model = SimilarityPredictions(collaborative_embds, similarity_metric="cosine")
        log("Info", "###### End predicting of collaborative similarity ######")

    def show_similar_movies_collaboration(self, item=1, top=20):
        similar_item = self.sim_model.predict_similar_items(seed_item=item, n=top)
        log("Info", "###### Similarity {} ".format(similar_item))
        # items = self.get_movie_ids(similar_item['item_id'].values())
        items = similar_item['item_id'].values()
        log("Info", "###### Similar Items {} ".format(items))
        log("Info",
            "###### Similar Movies to {} are {}".format(item, self.movies.loc[self.movies['movieId'].isin(items)]))


if __name__ == "__main__":
    cwd = os.getcwd()
    ## methods are [svd, ae]
    app = Main()
    app.get_dataframes()


    # app.collaborative_emb('ae')
    app.collaborative_sim('ae')
    app.show_similar_movies_collaboration(item=1, top=5)


    # app.content_emd('ae')
    # app.content_sim('ae')
    # app.show_similar_movies_content(item=1, top=5)
