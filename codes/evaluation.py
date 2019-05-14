import ipdb
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


def plot_item_value_count(df):
    counts = df['movieId'].value_counts()
    counts.plot(kind='bar')
    # ipdb.set_trace()
    # plt.xticks('off')
    # plt.title('Histogram of IQ')
    frame1 = plt.gca()
    # frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_xaxis().set_ticks([])
    plt.ylabel('Number of Hits')
    plt.xlabel('MovieId')
    # plt.grid(True)
    plt.savefig("result/imgs/plot.png", bbox_inches='tight')
    plt.show()

def RMSE(result):
    err = 0
    for item in result:
        err += (item[0] - item[1]) ** 2
    return sqrt(err / len(result))

def MAE(result):
    err = 0
    for item in result:
        err += (item[0] - item[1])
    return err / len(result)