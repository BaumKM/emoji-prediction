from typing import Dict, List

import numpy as np
import sklearn as sk
import tensorflow as tf

import plotting

BATCH_SIZE = 50


def evaluate_model(model: tf.keras.Model, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray,
                   y_test: np.ndarray, training_history: Dict[str, List[float]]):
    """
        Evaluates the trained model and displaying various metrics and plots.
        :param model: trained model
        :param x_train: The input training data.
        :param x_test: The input testing data.
        :param y_train: The target training labels.
        :param y_test: The target testing data.
        :param training_history: dictionary, that contains the training history
        :return: None
    """

    # Evaluate model on train and test data
    train_evaluation = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
    test_evaluation = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    # Get predictions and actual labels
    train_prediction = get_most_likely_label(model.predict(x_train, batch_size=BATCH_SIZE))
    test_prediction = get_most_likely_label(model.predict(x_test, batch_size=BATCH_SIZE))
    actual_train_label = get_most_likely_label(y_train)
    actual_test_label = get_most_likely_label(y_test)

    # Plot accuracy and loss
    plotting.plot_accuracy(training_history, model.name)
    plotting.plot_loss(training_history, model.name)

    # Plot confusion matrices
    plotting.plot_confusion_matrix(actual_train_label, train_prediction, f"{model.name}_train")
    plotting.plot_confusion_matrix(actual_test_label, test_prediction, f"{model.name}_test")

    # Print model summary
    plotting.print_model(model)

    # Calculate precision, recall, and F1 scores
    precision = sk.metrics.precision_score(actual_test_label, test_prediction, average="macro")
    recall = sk.metrics.recall_score(actual_test_label, test_prediction, average="macro")
    f1 = sk.metrics.f1_score(actual_test_label, test_prediction, average="macro")
    f1_detailed = sk.metrics.f1_score(actual_test_label, test_prediction, average=None)

    # Print metrics
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print("F1 Score (Detailed):", f1_detailed)


def get_most_likely_label(prediction: np.ndarray):
    """
       Gets the most likely prediction, based on a probability distribution.
       :param prediction: the prediction of the model
       :return: most likely label
   """
    return np.argmax(prediction, axis=1)
