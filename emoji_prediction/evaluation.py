import numpy as np
import sklearn as sk

import plotting

BATCH_SIZE = 50


def evaluate_model(model, x_train, x_test, y_train, y_test, training_history):
    train_evaluation = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
    # also evaluate test results because the progress bar shows the mean over the batches
    test_evaluation = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    train_prediction = get_most_likely_label(model.predict(x_train, batch_size=BATCH_SIZE))
    test_prediction = get_most_likely_label(model.predict(x_test, batch_size=BATCH_SIZE))
    actual_train_label = get_most_likely_label(y_train)
    actual_test_label = get_most_likely_label(y_test)

    plotting.plot_accuracy(training_history, model.name)
    plotting.plot_loss(training_history, model.name)
    plotting.plot_confusion_matrix(actual_train_label, train_prediction, f"{model.name}_train")
    plotting.plot_confusion_matrix(actual_test_label, test_prediction, f"{model.name}_test")
    plotting.print_model(model)

    precision = sk.metrics.precision_score(actual_test_label, test_prediction, average="macro")
    recall = sk.metrics.recall_score(actual_test_label, test_prediction, average="macro")
    f1 = sk.metrics.f1_score(actual_test_label, test_prediction, average="macro")
    f1_detailed = sk.metrics.f1_score(actual_test_label, test_prediction, average=None)
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")


def get_most_likely_label(prediction):
    return np.argmax(prediction, axis=1)
