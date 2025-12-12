import argparse
import sys
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import collections
from sklearn.cluster import KMeans
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def setup_arguments():
    parser = argparse.ArgumentParser(prog="kmeans", description="Handles K-Means clustering for classification AI.")
    parser.add_argument('input', help="Input data in single-column CSV format.")
    parser.add_argument('-v', '--verbose', help="Show each k-means algorithm run.", action='store_true')
    parser.add_argument('-t', '--tokenise', help="Tokenises the provided CSV", action='store_true')
    parser.add_argument('-o', '--output', help="Outputs contents to a file.")
    parser.add_argument('-cc', '--clear-cache', help="Clears the NLTK download cache", action='store_true')
    return parser.parse_args(None)

def tokenise(df : pd.DataFrame, args):
    word_count = []
    lemmatizer = WordNetLemmatizer()
    for review in list(df['review'] ):
        sentence = nltk.sent_tokenize(review)
        words = nltk.word_tokenize(review)
        for word in words:
            base_word = lemmatizer.lemmatize(word)
            word_count.append(base_word)
    return word_count

def sentiment_analysis(args, dataset : list[str]):
    sid = SentimentIntensityAnalyzer()
    positive_words = []
    negative_words = []
    neutral_words = []
    
    for word in dataset:
        if sid.polarity_scores(word)['compound'] >= 0.5:
            positive_words.append(word)
        elif sid.polarity_scores(word)['compound'] <= -0.5:
            negative_words.append(word)
        else:
            neutral_words.append(word)
    return (positive_words, negative_words, neutral_words)
         
def open_csv(filename : str):
    if not filename.endswith('.csv'):
        return pd.DataFrame()
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    return df

def download_nltk_packages(args):
    nltk.data.clear_cache() if args.clear_cache else False

    nltk_path = os.path.join(os.getcwd(), "nltk_data/")
    os.makedirs(nltk_path, exist_ok=True)
    nltk.data.path = [nltk_path]

    packages = ['punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'punkt', 'vader_lexicon']
    for package in packages:
        nltk.download(package, download_dir=nltk_path)

# TODO verbose messages.
# TODO Logging.

if __name__ == "__main__":
    args = setup_arguments()
    download_nltk_packages(args)
    df = open_csv(args.input)
    if df.empty:
        print(f"[ERROR] {args.input} does not exist or isn't a csv file!")
        sys.exit(1)
    if args.tokenise:
        words = tokenise(df, args)
        (neg, pos, neu) = sentiment_analysis(args, words)
        plt.bar(["Negative", "Positive"], [len(neg), len(pos)])
        plt.title("Sentiment Analysis")
        plt.xlabel("Words")
        plt.ylabel("Occurence")
        plt.show()