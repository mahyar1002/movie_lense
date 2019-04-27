from sklearn.feature_extraction.text import TfidfVectorizer
from authoencoder import AutoEncoder
import pandas as pd
from helper import log, save_embeddings, save_data
from sklearn.decomposition import TruncatedSVD
import ipdb


class Tfidf():
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.ids = dataframe.index.tolist()
        # self.ids = dataframe.movieId.tolist()

    def tfidf_tokenizer(self, min_df, ngram_range, documents_column_name):
        tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english')
        # print(tfidf.vocabulary_)
        tfidf_matrix = tfidf.fit_transform(self.dataframe[documents_column_name])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=self.ids)
        return tfidf_df, tfidf_matrix

    def get_svd_embeddings(self, feature_matrix, n):
        svd = TruncatedSVD(n_components=n)
        latent_matrix = svd.fit_transform(feature_matrix)
        latent_df = pd.DataFrame(latent_matrix, index=self.ids)
        return latent_df

    def get_authoencoder_embedings(self, tfidf_df):
        ae = AutoEncoder(tfidf_df, validation_perc=0.1, lr=1e-3, intermediate_size=1000, encoded_size=100,
                         cuda_enable=False)
        ae.train_loop(epochs=15)
        losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)), columns=['train_loss', 'validation_loss'])
        losses['epoch'] = (losses.index + 1) / 3
        encoded = ae.get_encoded_representations()
        return losses, encoded

    def apply(self, method='svd'):
        tfidf_df, tfidf_matrix = self.tfidf_tokenizer(min_df=0.0001, ngram_range=(0, 1),
                                                      documents_column_name='document')
        log("Info", "shape: {}".format(tfidf_df.shape))
        log("Info", "\n{}".format(tfidf_df.head(10)))
        save_embeddings(tfidf_df, 'result/tfidf_df.pkl', 'pickle')


        if method == 'svd':
            latent_df = self.get_svd_embeddings(tfidf_matrix, 100)
            log("Info", "shape: {}".format(latent_df.shape))
            log("Info", "\n{}".format(latent_df.head(5)))
        elif method == 'ae':
            losses, latent_df = self.get_authoencoder_embedings(tfidf_df)
            log("Info", "shape: {}".format(latent_df.shape))
            save_embeddings(losses, 'result/content_losses.pkl', 'pickle')
        else:
            raise AssertionError("Please specify a correct method [svd, ae]")

        save_data(latent_df, 'result/content_embeddings_{}.pkl'.format(method))
