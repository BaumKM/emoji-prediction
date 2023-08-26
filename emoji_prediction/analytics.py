import numpy as np


def analyze_tweets(tweets: np.ndarray) -> (np.ndarray, int):
    """
    Analyzes the given dataset of tweets, to extract the unique vocabulary and maximum tweet length.

    :param tweets: An array of strings, each representing a tweet.
    :return: A tuple with two elements:
             - the unique vocabulary of the dataset
             - the maximum tweet length
    """
    vocabulary = []
    maximum_length = 0
    for row in range(tweets.shape[0]):
        words = tweets[row].split()
        maximum_length = max(maximum_length, len(words))
        vocabulary += words
    unique_vocabulary = np.unique(vocabulary)

    return unique_vocabulary, maximum_length


def analyze_labels(labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Analyzes the given labels of the dataset, by computing the label distribution.

    :param labels: The labels of the dataset.
    :return: A tuple with two elements:
             - array that contains the unique values
             - array that contains the number of occurrences
    """
    label_distribution = np.unique(labels, return_counts=True)
    return label_distribution


def create_cumulative_length_distribution(tweets: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Creates the cumulative length distribution of the dataset. The i'th value of the distribution contains the number
    of tweets, that have a length which is greater or equal to values[i]. values[i] contains the sorted unique tweet
    lengths.
    :param tweets: The tweets of the dataset.
    :return: Tuple with two elements:
             - unique lengths in the dataset
             - cumulative length distribution
    """
    length_distribution = create_length_distribution(tweets)
    values, counts = np.unique(length_distribution, return_counts=True)

    # the i-th element of the list contains the number of sentences that have a length that is >= values[i]
    cumulative_length = np.flip(np.cumsum(np.flip(counts)))
    return values, cumulative_length


def create_length_distribution(tweets: np.ndarray) -> list[int]:
    """
    Creates the length distribution of the dataset.
    :param tweets: Tweets of the dataset.
    :return: length distribution
    """
    return [len(sentence.split()) for sentence in tweets]
