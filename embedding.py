from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

GLOVE_PATH = 'resources/glove/glove.6B.50d.txt'


def create_text_encoding(dataset: np.ndarray,
                         vocabulary: np.ndarray,
                         output_length: int) -> (np.ndarray, List[str]):
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


def create_label_encoding(labels: np.ndarray) -> np.ndarray:
    return pd.get_dummies(labels, dtype=int).to_numpy()


def load_glove_embedding() -> dict[str, np.ndarray]:
    file = open(GLOVE_PATH, 'r', encoding='utf8')
    embedding = np.array(file.readlines())
    file.close()
    embedding = np.array(list(map(str.split, embedding)))
    embedding = dict(zip(embedding[:, 0], embedding[:, 1:]))
    return embedding


def create_embedding_matrix(decoder: List[str], output_length: int = 50) -> (np.ndarray, int):
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
