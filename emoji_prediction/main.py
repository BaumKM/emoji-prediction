import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

import dataset
import plotting
from emoji_prediction import analytics, embedding

SEED = 3
TEST_SIZE = 0.3


def train_rnn(tweets: np.ndarray, labels: np.ndarray):
    y = embedding.create_label_encoding(labels)
    categorical_count = y.shape[1]

    vocabulary, maximum_length = analytics.analyze_tweets()
    x, decoder = embedding.create_text_encoding(tweets, vocabulary, maximum_length)

    embedding_matrix, count = embedding.create_embedding_matrix(decoder)
    model = create_rnn_model(embedding_matrix, categorical_count)

    x_train, x_test, y_train, y_test = split_data(x, y)

    history = model.fit(x_train, y_train, batch_size=5, epochs=100, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test, batch_size=50)
    plotting.plot_accuracy(history.history, "rnn")
    plotting.plot_loss(history.history, "rnn")
    test_prediction = model.predict(x_test, batch_size=50)
    train_prediction = model.predict(x_train, batch_size=50)
    plotting.plot_confusion_matrix(y_test, test_prediction, "rnn_test")
    plotting.plot_confusion_matrix(y_train, train_prediction, "rnn_train")


def train_fnn(tweets: np.ndarray, labels: np.ndarray):
    y = embedding.create_label_encoding(labels)
    categorical_count = y.shape[1]

    vocabulary, maximum_length = analytics.analyze_tweets()
    x, decoder = embedding.create_text_encoding(tweets, vocabulary, maximum_length)

    embedding_matrix, count = embedding.create_embedding_matrix(decoder)
    model = create_fnn_model(embedding_matrix, maximum_length, categorical_count, 0.01)
    x_train, x_test, y_train, y_test = split_data(x, y)

    history = model.fit(x_train, y_train, batch_size=5, epochs=100, validation_data=(x_test, y_test))

    # evaluate test results because the progress bar shows the mean over the batches
    test_evaluation = model.evaluate(x_test, y_test, batch_size=50)
    train_evaluation = model.evaluate(x_train, y_train, batch_size=50)

    test_prediction = model.predict(x_test, batch_size=50)
    train_prediction = model.predict(x_train, batch_size=50)
    plotting.plot_accuracy(history.history, "fnn")
    plotting.plot_loss(history.history, "fnn")
    plotting.plot_confusion_matrix(y_test, test_prediction, "fnn_test")
    plotting.plot_confusion_matrix(y_train, train_prediction, "fnn_train")
    plotting.print_model(model)


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
                  metrics=["categorical_accuracy", "accuracy", "categorical_crossentropy"])
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
