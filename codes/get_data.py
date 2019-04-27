import numpy as np
import pandas as pd
from helper import log, save_embeddings, save_data
import os
import ipdb


def tags_to_panda(directory):
    tags = pd.read_csv(directory).drop(["timestamp"], axis=1)

    log('Info', "########## Tags Summary ##########")
    log('Info', "{} unique movies in tags.csv".format(len(tags.movieId.unique())))
    log('Info', "shape: {}".format(tags.shape))
    log('Info', '\n{}'.format(tags.sample(5)))
    log('Info', "########## End of Tags Summary ##########")

    tags.fillna("", inplace=True)
    tags['tag'] = tags['tag'].str.lower()
    tags = pd.DataFrame(tags.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
    tags.reset_index(inplace=True)
    log("Info", "########## Merging Tags ##########")
    log("Info", "There are {} unique movies".format(len(tags.movieId)))
    log("Info", "\n{}".format(tags.head(5)))
    log("Info", "########## End of Merging Tags ##########")

    return tags


def rating_to_panda(directory):
    ratings = pd.read_csv(directory).drop(["timestamp"], axis=1)

    log('Info', "###### Ratings Summary ######")
    log('Info', "shape: {}".format(ratings.shape))
    log('Info', '\n{}'.format(ratings.sample(3)))
    log('Info', "###### End of Ratings Summary ######")

    return ratings


def movies_to_panda(directory):
    movies = pd.read_csv(directory)
    movies['genres'] = movies['genres'].str.lower()
    movies['genres'] = movies['genres'].str.replace(pat="|", repl=" ")
    movies['genres'] = movies['genres'].str.replace(pat="-", repl="")

    log('Info', "###### Movies Summary ######")
    log('Info', "{} unique movies in movies.csv".format(len(movies.movieId.unique())))
    log('Info', "shape: {}".format(movies.shape))
    log('Info', '\n{}'.format(movies.head(5)))
    log('Info', "###### End of Movies Summary ######")

    return movies


def merge_movies_tags(movies, tags):
    merged = pd.merge(movies, tags, on="movieId", how="left")
    # merged['tag'] = merged["genres"] + ' ' + merged["tag"].map(str)
    merged['document'] = merged['genres'].str.cat(merged['tag'], sep=' ', na_rep="").str.strip()
    merged.drop(['tag', 'genres', 'title'], axis=1, inplace=True)
    merged.fillna("", inplace=True)

    log('Info', "###### Merged Movie & Tags Summary ######")
    log('Info', "{} unique movies in merged".format(len(merged.movieId.unique())))
    log('Info', "shape: {}".format(merged.shape))
    log('Info', '\n{}'.format(merged.sample(10)))
    log('Info', "###### End of Merged Movie & Tags Summary ######")

    # merged.query("movieId == 5899").document.values

    return merged
