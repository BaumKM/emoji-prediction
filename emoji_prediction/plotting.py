import os

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn as sk
from keras_visualizer import visualizer
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tensorflow import keras

import analytics

dir_name = os.path.dirname(__file__)

CONFUSION_PATH = os.path.join(dir_name, "resources/graphics/confusion/{name}_confusion.eps")
LOSS_PATH = os.path.join(dir_name, "resources/graphics/loss/{name}_loss.eps")
ACCURACY_PATH = os.path.join(dir_name, "resources/graphics/accuracy/{name}_accuracy.eps")
SENTENCE_LENGTH_PATH = os.path.join(dir_name, "resources/graphics/dataset/cumulative_sentence_length.eps")
CUMULATIVE_LENGTH_PATH = os.path.join(dir_name, "resources/graphics/dataset/dataset/cumulative_length.eps")
FNN_DETAILED_ARCHITECTURE_PATH = os.path.join(dir_name, "resources/graphics/structure/fnn_detailed")
FFN_ARCHITECTURE_PATH = os.path.join(dir_name, "resources/graphics/structure/fnn.png")

image_names = ["red-heart", "baseball", "happy", "disappointed", "dishes"]


def print_model(model):
    visualizer(model, file_name=FNN_DETAILED_ARCHITECTURE_PATH, file_format='png', view=False)
    keras.utils.plot_model(model, to_file=FFN_ARCHITECTURE_PATH, show_shapes=True, show_layer_names=True)


def plot_accuracy(history: dict[str, list], name):
    fig, ax = create_simple_figure()
    x_values = np.arange(1, len(history['accuracy']) + 1)

    ax.set_ylabel("accuracy", labelpad=8)
    ax.set_xlabel("epoch", labelpad=8)

    ax.plot(x_values, history['accuracy'], label="training data")
    ax.plot(x_values, history['val_accuracy'], label="test data")

    val_maximum_pos = np.argmax(history['val_accuracy']) + 1
    val_maximum = round(np.max(history['val_accuracy']), 2)

    bbox = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrow = dict(arrowstyle="->", connectionstyle="arc", color="blue", lw=3)
    kw = dict(xycoords='data', textcoords="data", bbox=bbox, ha="left", va="bottom")
    ax.annotate(f"epoch: {val_maximum_pos}, accuracy: {val_maximum}", xy=(val_maximum_pos, val_maximum),
                arrowprops=arrow,
                xytext=(val_maximum_pos - 10, val_maximum + 0.1), **kw)

    ax.legend()
    fig.tight_layout()
    save_figure(ACCURACY_PATH.format(name=name))


def plot_loss(history: dict[str, list], name):
    fig, ax = create_simple_figure()

    x_values = np.arange(1, len(history['loss']) + 1)
    ax.set_ylabel("loss", labelpad=8, )
    ax.set_xlabel("epoch", labelpad=8)

    ax.plot(x_values, history['categorical_crossentropy'], label="training data")
    ax.plot(x_values, history['val_categorical_crossentropy'], label="test data")

    val_minimum_pos = np.argmin(history['val_categorical_crossentropy']) + 1
    val_minimum = round(np.min(history['val_categorical_crossentropy']), 2)

    bbox = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrow = dict(arrowstyle="->", connectionstyle="arc", color="blue", lw=3)
    kw = dict(xycoords='data', textcoords="data", bbox=bbox, ha="left", va="bottom")
    ax.annotate(f"epoch: {val_minimum_pos}, loss: {val_minimum}", xy=(val_minimum_pos, val_minimum), arrowprops=arrow,
                xytext=(val_minimum_pos + 10, val_minimum - 0.25), **kw)
    ax.legend()
    fig.tight_layout()
    save_figure(LOSS_PATH.format(name=name))


def plot_cumulative_length():
    fig, ax = create_simple_figure()
    values, cumulative_length = analytics.create_cumulative_length_distribution()
    total_elements = cumulative_length[0]
    relative_cumulative_length = np.around(np.divide(cumulative_length, total_elements), 2)

    ax.set_ylabel("Relative Frequency", labelpad=20)
    ax.set_xlabel("Tweet Length", labelpad=15)

    bars = ax.bar(values, relative_cumulative_length, width=0.4)
    ax.bar_label(bars, relative_cumulative_length)
    ax.set_xticks(values)
    fig.tight_layout()
    save_figure(CUMULATIVE_LENGTH_PATH)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name):
    true_label = np.argmax(y_true, axis=1)
    predicted_label = np.argmax(y_pred, axis=1)

    confusion_matrix = sk.metrics.confusion_matrix(true_label, predicted_label)

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


def create_simple_figure():
    plt.style.use("bmh")
    font = {
        'size': 13}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    ax.yaxis.get_label().set_fontsize(15)
    ax.xaxis.get_label().set_fontsize(15)

    return fig, ax


def save_figure(path):
    plt.savefig(path, dpi=1200)


if __name__ == '__main__':
    plot_cumulative_length()
