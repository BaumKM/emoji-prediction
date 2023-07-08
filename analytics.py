import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras_visualizer import visualizer
from tensorflow import keras


def analyze_tweets(tweets: np.ndarray) -> (np.ndarray, int):
    vocabulary = []
    maximum_length = 0
    for row in range(tweets.shape[0]):
        words = tweets[row].split()
        maximum_length = max(maximum_length, len(words))
        vocabulary += words
    unique_vocabulary = list(dict.fromkeys(vocabulary))
    return np.array(unique_vocabulary), maximum_length


def analyze_labels(labels: np.ndarray):
    result = [0 for x in range(labels.shape[1])]
    for row in range(labels.shape[0]):
        label = np.argmax(labels[row])
        result[label] += 1
    return result


def print_model(model):
    visualizer(model, file_name='test', file_format='png', view=True)
    #keras.utils.plot_model(model, to_file='graphics/fnn', show_shapes=True, show_layer_names=True)


def plot_loss(history: dict[str, list]):
    font = {
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    x_values = np.arange(len(history['accuracy']))
    ax.set_ylabel("accuracy", labelpad=8, weight='bold')
    ax.set_xlabel("epoch", labelpad=8, weight='bold')
    ax.yaxis.get_label().set_fontsize(15)
    ax.xaxis.get_label().set_fontsize(15)


    ax.plot(x_values, history['accuracy'], label="training data")
    ax.plot(x_values, history['val_accuracy'], label="test data")
    ax.annotate(str(round(history['accuracy'][-1], 2)),
                (x_values[-1], history['accuracy'][-1]),
                (x_values[-1]-0.5, history['accuracy'][-1]-0.02))
    ax.annotate(str(round(history['val_accuracy'][-1], 2)),
                (x_values[-1], history['val_accuracy'][-1]),
                (x_values[-1]-0.5, history['val_accuracy'][-1]+0.01))
    ax.legend()
    ax.set_title("Model Accuracy", fontsize=15, weight='bold')
    plt.show()


def plot_label_distribution(distribution):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    x_values = np.arange(len(distribution))
    total = np.sum(distribution)
    ax.set_title("Label Distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Label")
    ax.set_xticks(x_values, x_values)
    # for i, v in enumerate(distribution):
    #     percentage = (v/total)
    #     ax.text(i-0.15, v + 1, str(round(percentage, 2)))
    bars = ax.bar(x_values, distribution, width=0.4)
    percentages = [str(round((x/total)*100)) + "%" for x in distribution]

    ax.bar_label(bars, percentages)
    plt.show()

