import numpy as np
import sklearn.model_selection
import tensorflow as tf
from pandas import DataFrame
from tensorflow import keras

import analytics
import embedding
import preprocessor

SEED = 3
TEST_SIZE = 0.3


def train_rnn(dataset: DataFrame):
    tweets = dataset.iloc[:, 0].values
    labels = dataset.iloc[:, 1].values
    y = embedding.create_label_encoding(labels)
    categorical_count = y.shape[1]

    vocabulary, maximum_length = analytics.analyze_tweets(tweets)
    x, decoder = embedding.create_text_encoding(tweets, vocabulary, maximum_length)

    embedding_matrix, count = embedding.create_embedding_matrix(decoder)
    model = create_rnn_model(embedding_matrix, categorical_count)

    x_train, x_test, y_train, y_test = split_data(x, y)

    history = model.fit(x_train, y_train, batch_size=5, epochs=65, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test, batch_size=50)
    analytics.plot_accuracy(history.history)
    analytics.plot_loss(history.history)
    # analytics.print_model(model)


def train_fnn(dataset: DataFrame):
    tweets = dataset.iloc[:, 0].values
    labels = dataset.iloc[:, 1].values
    y = embedding.create_label_encoding(labels)
    categorical_count = y.shape[1]

    vocabulary, maximum_length = analytics.analyze_tweets(tweets)
    x, decoder = embedding.create_text_encoding(tweets, vocabulary, maximum_length)

    embedding_matrix, count = embedding.create_embedding_matrix(decoder)
    model = create_fnn_model(embedding_matrix, maximum_length, categorical_count, 0.01)
    x_train, x_test, y_train, y_test = split_data(x, y)
    history = model.fit(x_train, y_train, batch_size=5, epochs=38, validation_data=(x_test, y_test))

    distribution_train = analytics.analyze_labels(y_train)
    distribution_test = analytics.analyze_labels(y_test)

    model.evaluate(x_test, y_test, batch_size=50)
    print(distribution_test)
    print(distribution_train)
    analytics.plot_accuracy(history.history)
    analytics.plot_loss(history.history)
    # analytics.plot_label_distribution(analytics.analyze_labels(y))


def create_rnn_model(embedding_matrix, categorical_count):
    regularization = 0
    model = keras.Sequential()
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
                  metrics=["accuracy"])
    print(model.summary())
    return model


def create_fnn_model(embedding_matrix, time_steps, categorical_count: int,
                     regularization: float):
    model = keras.Sequential()
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
                  metrics=["accuracy"])
    print(model.summary())
    return model


def split_data(x: np.ndarray, y: np.ndarray):
    return sklearn.model_selection.train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)


def setup_tensorflow():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


if __name__ == '__main__':
    setup_tensorflow()
    data = preprocessor.load_dataset()
    # train_fnn(data)
    train_rnn(data)
