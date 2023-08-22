import os
import re

import numpy as np
import pandas as pd
from pandas import DataFrame

dir_name = os.path.dirname(__file__)
RAW_DATASET_PATH = os.path.join(dir_name, "resources/datasets/emoji_data.csv")
CLEAN_DATASET_PATH = os.path.join(dir_name, "resources/datasets/cleaned_data.csv")
LABEL_MAPPING_PATH = os.path.join(dir_name, "resources/datasets/label_mapping.csv")

AT_REGEX = r"@\S*"
HTML_REGEX = r"https?://\S+"
SPECIAL_CHAR_REGEX = r"[^a-z\s]"


def __clean_dataset() -> DataFrame:
    """
    loads the raw data file as a Dataframe and cleans it
    :return: cleaned dataset
    """
    raw_dataset = pd.read_csv(RAW_DATASET_PATH, header=None)
    tweets = raw_dataset.iloc[:, 0]
    tweets = tweets.apply(__clean_tweet)
    raw_dataset.iloc[:, 0] = tweets
    return raw_dataset


def __clean_tweet(tweet: str) -> str:
    tweet = str(tweet)
    tweet = tweet.lower()
    tweet = re.sub(SPECIAL_CHAR_REGEX, '', tweet)

    words = tweet.split()
    return " ".join(words)


def __save_dataset(dataset: DataFrame):
    dataset.to_csv(CLEAN_DATASET_PATH, index=False, header=False)


def load_dataset() -> (np.ndarray, np.ndarray):
    if not os.path.exists(CLEAN_DATASET_PATH):
        print("Loading dataset")
        data = __clean_dataset()
        __save_dataset(data)
    else:
        data = pd.read_csv(CLEAN_DATASET_PATH, header=None)
    return data.iloc[:, 0].values, data.iloc[:, 1].values


def load_label_mapping() -> np.ndarray:
    data = pd.read_csv(LABEL_MAPPING_PATH, header=None)
    return data.values


if __name__ == '__main__':
    load_dataset()
