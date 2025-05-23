from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import ipdb

class SimilarityPredictions(object):
    def __init__(self, embeddings, similarity_metric='cosine'):
        assert similarity_metric in ['cosine', 'euclidean'], "unsupported similarity metric."
        self.embeddings = embeddings
        self.ids = embeddings.index.tolist()
        self.similarity_metric = similarity_metric
        if similarity_metric == 'cosine':
            self.similarity_matrix = self.calculate_cosine_similarity_matrix()
        if similarity_metric == 'euclidean':
            self.similarity_matrix = self.calculate_euclidean_distances_matrix()

    def calculate_cosine_similarity_matrix(self):
        similarity_matrix = pd.DataFrame(cosine_similarity(
            X=self.embeddings),
            index=self.ids)
        similarity_matrix.columns = self.ids
        return similarity_matrix

    def calculate_euclidean_distances_matrix(self):
        similarity_matrix= pd.DataFrame(euclidean_distances(
            X=self.embeddings),
            index=self.ids)
        similarity_matrix.columns = self.ids
        return similarity_matrix

    def predict_similar_items(self, seed_item, n, method):
        similar_items = pd.DataFrame(self.similarity_matrix.loc[seed_item])
        col_name = "similarity_score_{}".format(method)
        similar_items.columns = [col_name]
        if self.similarity_metric == 'cosine':
            similar_items = similar_items.sort_values(col_name, ascending=False)
        if self.similarity_metric == 'euclidean':
            similar_items = similar_items.sort_values(col_name, ascending=True)
        similar_items = similar_items.head(n)
        similar_items.reset_index(inplace=True)
        similar_items = similar_items.rename(index=str, columns={"index": "item_id"})
        return similar_items
