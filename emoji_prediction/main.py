import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

import dataset
import evaluation
from emoji_prediction import analytics, embedding

SEED = 3
TEST_SIZE = 0.3


def train_rnn(tweets: np.ndarray, labels: np.ndarray):
    name = "rnn"
    y = embedding.create_label_encoding(labels)
    categorical_count = y.shape[1]

    vocabulary, maximum_length = analytics.analyze_tweets()
    x, decoder = embedding.create_text_encoding(tweets, vocabulary, maximum_length)

    embedding_matrix, count = embedding.create_embedding_matrix(decoder)
    model = create_rnn_model(embedding_matrix, categorical_count, name=name)

    x_train, x_test, y_train, y_test = split_data(x, y)

    history = model.fit(x_train, y_train, batch_size=5, epochs=100, validation_data=(x_test, y_test))
    evaluation.evaluate_model(model, x_train, x_test, y_train, y_test, history.history)


def train_fnn(tweets: np.ndarray, labels: np.ndarray):
    name="fnn"
    y = embedding.create_label_encoding(labels)
    categorical_count = y.shape[1]

    vocabulary, maximum_length = analytics.analyze_tweets()
    x, decoder = embedding.create_text_encoding(tweets, vocabulary, maximum_length)

    embedding_matrix, count = embedding.create_embedding_matrix(decoder)
    model = create_fnn_model(embedding_matrix, maximum_length, categorical_count, 0.01, name)
    x_train, x_test, y_train, y_test = split_data(x, y)

    history = model.fit(x_train, y_train, batch_size=5, epochs=37, validation_data=(x_test, y_test))
    evaluation.evaluate_model(model, x_train, x_test, y_train, y_test, history.history)




def create_rnn_model(embedding_matrix, categorical_count, name):
    regularization = 0
    model = keras.Sequential(name=name)
    model.add(keras.layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                     trainable=False,
                                     embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                     mask_zero=True))
    model.add(keras.layers.SimpleRNN(6, return_sequences=True))
    model.add(keras.layers.SimpleRNN(5, return_sequences=False,
                                     kernel_regularizer=keras.regularizers.L2(l2=regularization),
                                     bias_regularizer=keras.regularizers.L2(l2=regularization)))
    model.add(keras.layers.Dense(categorical_count, activation='softmax',
                                 kernel_regularizer=keras.regularizers.L2(l2=regularization),
                                 bias_regularizer=keras.regularizers.L2(l2=regularization)))

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  metrics=["categorical_accuracy", "accuracy", "categorical_crossentropy"])
    print(model.summary())
    return model


def create_fnn_model(embedding_matrix, time_steps, categorical_count: int,
                     regularization: float, name):
    model = keras.Sequential(name=name)
    model.add(keras.layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                     input_length=time_steps,
                                     trainable=False,
                                     embeddings_initializer=keras.initializers.Constant(embedding_matrix)))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(8, activation='relu',
                                 kernel_regularizer=keras.regularizers.L2(l2=regularization),
                                 bias_regularizer=keras.regularizers.L2(l2=regularization)))

    model.add(keras.layers.Dense(7, activation='relu',
                                 kernel_regularizer=keras.regularizers.L2(l2=regularization),
                                 bias_regularizer=keras.regularizers.L2(l2=regularization)))

    model.add(keras.layers.Dense(categorical_count, activation='softmax',
                                 kernel_regularizer=keras.regularizers.L2(l2=regularization),
                                 bias_regularizer=keras.regularizers.L2(l2=regularization)))
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  metrics=["categorical_accuracy", "accuracy", "categorical_crossentropy"])
    print(model.summary())
    return model


def split_data(x: np.ndarray, y: np.ndarray):
    return sklearn.model_selection.train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)


def setup_tensorflow():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


if __name__ == '__main__':
    setup_tensorflow()
    dataset = dataset.load_dataset()

    train_fnn(dataset[0], dataset[1])
    setup_tensorflow()
    train_rnn(dataset[0], dataset[1])
