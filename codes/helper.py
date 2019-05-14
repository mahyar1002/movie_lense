from datetime import datetime
import pandas as pd
import pickle


def log(type, text):
    log_text = "{}: {} [at: {}]".format(type.upper(), text, datetime.now())
    file = open('log/log.txt', 'a')
    print(log_text)
    file.write('{}\n'.format(log_text))


def load_df(path):
    # log('Info', "###### Loading df from file: {} ######".format(path))
    return pd.DataFrame(pd.read_pickle('result/{}'.format(path)))

def load_data(path):
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    return data

def save_embeddings(df, path, file_format='csv'):
    # log('Info', "###### Saving embeddings to file: {} ######".format(path))
    assert file_format in ['csv', 'pickle'], "unsupported format"
    if file_format == 'csv':
        df.to_csv(path, header=True, index=True)
    elif file_format == 'pickle':
        df.to_pickle(path)


def save_data(data, path):
    # log('Info', "###### Saving data to file: {} ######".format(path))
    with open(path, 'wb') as fh:
        pickle.dump(data, fh)

# def movie_to_index(movies):
#     dict = {}
#     for id in movies:
#         dict[]
