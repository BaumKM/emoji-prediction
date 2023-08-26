import os
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

dir_name = os.path.dirname(__file__)
GLOVE_PATH = os.path.join(dir_name, 'resources/glove/glove.6B.50d.txt')


def create_text_encoding(dataset: np.ndarray,
                         vocabulary: np.ndarray,
                         output_length: int) -> (np.ndarray, List[str]):
    """
    Creates the text encoding for the given sentences, based on the vocabulary. Each word in each sentence is mapped
    to a single integer. Additionally, each sentence is padded with 0s to the output length.
    :param dataset: dataset with the sentences
    :param vocabulary: unique vocabulary
    :param output_length: target output length
    :return: encoded sentences padded to the output_length
    """
    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode='int',
        output_sequence_length=output_length,
        vocabulary=vocabulary)

    model = tf.keras.models.Sequential()
    model.add(keras.Input(shape=1, dtype=tf.string))
    model.add(vectorize_layer)
    encoded_data = model.predict(dataset, batch_size=50)
    decoder = vectorize_layer.get_vocabulary().copy()
    return encoded_data, decoder


def create_one_hot_encoding(labels: np.ndarray) -> np.ndarray:
    """
    Creates the one hot encoding for the labels.
    :param labels: labels of the dataset
    :return: one hot encoding
    """
    return pd.get_dummies(labels, dtype=int).to_numpy()


def load_glove_embedding() -> dict[str, np.ndarray]:
    """
    Loads the glove embedding as a dictionary which maps each string to its vector representation.
    :return: glove embedding dictionary
    """
    file = open(GLOVE_PATH, 'r', encoding='utf8')
    embedding = np.array(file.readlines())
    file.close()
    embedding = np.array(list(map(str.split, embedding)))
    embedding = dict(zip(embedding[:, 0], embedding[:, 1:]))
    return embedding


def create_embedding_matrix(decoder: List[str], output_length: int = 50) -> (np.ndarray, int):
    """
    Creates the tensorflow embedding matrix based on the encoded vocabulary. The i'th row of the embedding matrix
    contains the vector representation of the word with the encoding i.
    :param decoder: The decoder is a list which maps an integer to a word.
    :param output_length: The length of the vector representation. Default: 50
    :return: tensorflow embedding matrix
    """
    embedding = load_glove_embedding()
    embedding_matrix = np.zeros((len(decoder), output_length))
    unknown_words = 0
    for i, word in enumerate(decoder):
        embedded_word = embedding.get(word)
        if embedded_word is not None:
            embedding_matrix[i] = embedded_word
        else:
            unknown_words += 1
    return embedding_matrix, unknown_words
