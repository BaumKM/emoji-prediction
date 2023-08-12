
import seaborn as sns
import matplotlib
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

from keras_visualizer import visualizer
from tensorflow import keras


CONFUSION_PATH = "emoji_prediction/resources/graphics/{name}_confusion.png"

image_names = ["red-heart", "baseball", "happy", "disappointed", "dishes"]

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
    visualizer(model, file_name='emoji_prediction/resources/graphics/fnn_detailed', file_format='png', view=False)
    keras.utils.plot_model(model, to_file='emoji_prediction/resources/graphics/fnn.png', show_shapes=True, show_layer_names=True)


def plot_accuracy(history: dict[str, list]):
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
                (x_values[-1] - 0.5, history['accuracy'][-1] - 0.02))
    ax.annotate(str(round(history['val_accuracy'][-1], 2)),
                (x_values[-1], history['val_accuracy'][-1]),
                (x_values[-1] - 0.5, history['val_accuracy'][-1] + 0.01))
    ax.legend()
    ax.set_title("Model Accuracy", fontsize=15, weight='bold')
    plt.show()


def plot_loss(history: dict[str, list]):
    use_style()
    font = {
        'weight': 'bold',
        'size': 14}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    x_values = np.arange(len(history['loss']))
    ax.set_ylabel("loss", labelpad=8, weight='bold')
    ax.set_xlabel("epoch", labelpad=8, weight='bold')
    ax.yaxis.get_label().set_fontsize(15)
    ax.xaxis.get_label().set_fontsize(15)

    ax.plot(x_values, history['loss'], label="training data")
    ax.plot(x_values, history['val_loss'], label="test data")
    ax.annotate(str(round(history['loss'][-1], 2)),
                (x_values[-1], history['loss'][-1]),
                (x_values[-1] - 0.5, history['loss'][-1] + 0.07))
    ax.annotate(str(round(history['val_loss'][-1], 2)),
                (x_values[-1], history['val_loss'][-1]),
                (x_values[-1] - 0.5, history['val_loss'][-1] + 0.04))
    ax.legend()
    ax.set_title("Model Loss", fontsize=15, weight='bold')
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
    percentages = [str(round((x / total) * 100)) + "%" for x in distribution]

    ax.bar_label(bars, percentages)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name):
    true_label = np.argmax(y_true, axis=1)
    pred_label = np.argmax(y_pred, axis=1)

    confusion_matrix = sk.metrics.confusion_matrix(true_label, pred_label)

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, cmap="Blues", annot=True, ax=ax)

    ax.set_ylabel("True Label", labelpad=20)
    ax.set_xlabel("Predicted Label", labelpad=20)
    x_labels = ax.get_xticklabels()
    y_labels = ax.get_yticklabels()
    for x_label, y_label, image_name in zip(x_labels, y_labels, image_names):
        image = mpimg.imread(f"emoji_prediction/resources/emojis/{image_name}.png")
        create_tick_label(x_label.get_position()[0], 5, image, ax, (0, -7))
        create_tick_label(0, y_label.get_position()[1], image, ax, (-10, 0))

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    fig.tight_layout()
    plt.savefig(CONFUSION_PATH.format(name=name))


def create_tick_label(x, y, image, axes, box_offset):
    offset_image = OffsetImage(image, zoom=0.2)
    offset_image.image.axes = axes
    annotation_box = AnnotationBbox(offset_image, (x, y), xybox=box_offset, frameon=False,
                                    xycoords='data', boxcoords="offset points", pad=0)
    axes.add_artist(annotation_box)



def use_style():
    plt.style.use("bmh")
