import numpy as np

import dataset


def analyze_tweets() -> (np.ndarray, int):
    tweets = load_tweets()
    vocabulary = []
    maximum_length = 0
    for row in range(tweets.shape[0]):
        words = tweets[row].split()
        maximum_length = max(maximum_length, len(words))
        vocabulary += words
    unique_vocabulary = np.unique(vocabulary)

    return unique_vocabulary, maximum_length


def analyze_labels():
    labels = load_labels()
    result = [0 for x in range(labels.shape[1])]
    for row in range(labels.shape[0]):
        label = np.argmax(labels[row])
        result[label] += 1
    return result


def create_cumulative_length_distribution():
    length_distribution = create_length_distribution()
    values, counts = np.unique(length_distribution, return_counts=True)

    # the i-th element of the list contains the number of sentences that have a length that is >= values[i]
    cumulative_length = np.flip(np.cumsum(np.flip(counts)))
    return values, cumulative_length


def create_length_distribution():
    tweets = load_tweets()
    return [len(sentence.split()) for sentence in tweets]


def load_tweets():
    return dataset.load_dataset()[0]


def load_labels():
    return dataset.load_dataset()[1]
