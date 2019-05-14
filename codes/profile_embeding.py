from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ipdb
from helper import save_embeddings


class ProfileFiltering:
    def __init__(self, profile_df):
        self.profile_df = profile_df

    def pca_to_profile(self):
        x = StandardScaler().fit_transform(self.profile_df.values)
        pca = PCA(n_components=5)

        principalComponents = pca.fit_transform(x)

        profile_df = pd.DataFrame(data=principalComponents
                                  , columns=['f1', 'f2', 'f3', 'f4', 'f5'])

        save_embeddings(profile_df, 'result/profile_embeddings_pca.pkl', 'pickle')
