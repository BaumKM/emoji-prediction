import numpy as np


def analyze_tweets(tweets: np.ndarray) -> (np.ndarray, int):
    vocabulary = []
    maximum_length = 0
    for row in range(tweets.shape[0]):
        words = tweets[row].split()
        maximum_length = max(maximum_length, len(words))
        vocabulary += words
    unique_vocabulary = np.unique(vocabulary)

    return unique_vocabulary, maximum_length


def analyze_labels(labels: np.ndarray) -> (np.ndarray, np.ndarray):
    label_distribution = np.unique(labels, return_counts=True)
    return label_distribution


def create_cumulative_length_distribution(tweets: np.ndarray) -> (np.ndarray, np.ndarray):
    length_distribution = create_length_distribution(tweets)
    values, counts = np.unique(length_distribution, return_counts=True)

    # the i-th element of the list contains the number of sentences that have a length that is >= values[i]
    cumulative_length = np.flip(np.cumsum(np.flip(counts)))
    return values, cumulative_length


def create_length_distribution(tweets: np.ndarray) -> list[int]:
    return [len(sentence.split()) for sentence in tweets]
