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

def setup_arguments():
    parser = argparse.ArgumentParser(prog="kmeans", description="Handles K-Means clustering for classification AI.")
    parser.add_argument('input', help="Input data in single-column CSV format.")
    parser.add_argument('-v', '--verbose', help="Show each k-means algorithm run.", action='store_true')
    parser.add_argument('-c', '--count_words', help="Counts words in a sentence and outputs to the console or file.", action='store_true')
    parser.add_argument('-o', '--output', help="Outputs contents to a file.")
    parser.add_argument('-cc', '--clear-cache', help="Clears the NLTK download cache", action='store_true')
    return parser.parse_args()

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

    packages = ['punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'punkt']
    for package in packages:
        nltk.download(package, download_dir=nltk_path)

if __name__ == "__main__":
    args = setup_arguments()
    download_nltk_packages(args)
    df = open_csv(args.input)
    if df.empty:
        print(f"[ERROR] {args.input} does not exist or isn't a csv file!")
        sys.exit(0)
    if args.count_words:
        print(tokenise(df, args))