"""
Put your NetID here.
"""


import sys
import argparse
import numpy as np


class Opts:
    def __init__(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('train', default=None, help='The path to the training set file.')
        parser.add_argument('--test', default=None, help='The path to the test set file. If used, the model should be in the final state and no further hyper-parameter tuning is allowed.')
        parser.add_argument('-t', '--threshold', type=int, default=26, help='The threshold of the number of times to choose a word.')
        self.__dict__.update(parser.parse_args(argv).__dict__)


def read_data(filename):
    """
    Read the dataset from the file given by name $filename.
    The returned object should be a list of pairs of data, such as
        [
            (True , ['a', 'b', 'c']),
            (False, ['d', 'e', 'f']),
            ...
        ]
    """

    # Fill in your code here.

    pass


def split_train(original_train_data):
    """
    Split the original training set into two sets:
        * the training set, and
        * the validation set.
    """

    # Fill in your code here.

    pass


def create_wordlist(original_train_data, threshold=26):
    """
    Create a word list from the original training set.
    Only get a word if it appears in at least $threshold emails.
    """

    # Fill in your code here.

    pass


class Model:
    @staticmethod
    def count_labels(data):
        """
        Count the number of positive labels and negative labels.
        Returns (a tuple or a numpy array of two elements):
            * negative_count: a non-negative integer, which represents the number of negative labels;
            * positive_count: a non-negative integer, which represents the number of positive labels.
        """

        # Fill in your code here.

        pass

    @staticmethod
    def count_words(wordlist, label_counts, data):
        """
        Count the number of times that each word appears in emails under a given label.
        Returns (a numpy array):
            * negative_word_counts: a numpy array with shape (L, N-non-spam), where L is the length of $wordlist, N-non-spam is the number of non-spam emails, and
                - negative_word_counts[i, j] represents the number of times that word $wordlist[i] appears in the j-th non-spam email; and
            * positive_word_counts: a numpy array with shape (L, N-spam), where L is the length of $wordlist, N-spam is the number of spam emails, and
                - positive_word_counts[i, j] represents the number of times that word $wordlist[i] appears in the j-th spam email.
        """

        # Fill in your code here.

        pass

    @staticmethod
    def calculate_probability(label_counts, negative_word_counts, positive_word_counts):
        """
        Calculate the probabilities, both the prior and likelihood.
        Returns (a pair of numpy array):
            * prior_probs: a numpy array with shape (2, ), only two elements, where
                - prior_probs[0] is the prior probability of negative labels, and
                - prior_probs[1] is the prior probability of positive labels;
            * likelihood_mus: a numpy array with shape (2, L), where L is the length of the word list,
                - likelihood_mus[0, i] represents the mean value (mu) of the likelihood probability of the $i-th word in the word list, given that the email is non-spam (negative), and
                - likelihood_mus[1, i] represents the mean value (mu) of the likelihood probability of the $i-th word in the word list, given that the email is spam (positive); and
            * likelihood_sigmas: a numpy array with shape (2, L), where L is the length of the word list,
                - likelihood_sigmas[0, i] represents the deviation value (sigma) of the likelihood probability of the $i-th word in the word list, given that the email is non-spam (negative), and
                - likelihood_sigmas[1, i] represents the deviation value (sigma) of the likelihood probability of the $i-th word in the word list, given that the email is spam (positive).
        """

        # Fill in your code here.

        pass

    def __init__(self, wordlist):
        self.wordlist = wordlist

    def fit(self, data):
        label_counts = self.__class__.count_labels(data)
        negative_word_counts, positive_word_counts = self.__class__.count_words(self.wordlist, label_counts, data)

        self.prior_probs, self.likelihood_mus, self.likelihood_sigmas = self.__class__.calculate_probability(label_counts, negative_word_counts, positive_word_counts)

        # You may do some additional processing of variables here, if you want.

    def predict(self, x):
        """
        Predict whether email $x is a spam or not.
        Returns:
            * y: a boolean value indicating whether $x is a spam or not.
        """

        # Fill in your code here.

        pass


def main(argv):
    opts = Opts(argv)

    if opts.test is None:
        original_train_data = read_data(opts.train)
        train_data, val_data = split_train(original_train_data)
    else:
        original_train_data = read_data(opts.train)
        train_data = original_train_data
        val_data = read_data(opts.test)

    # Create the word list.
    wordlist = create_wordlist(original_train_data, opts.threshold)

    model = Model(wordlist)
    model.fit(train_data)

    error_count = sum([y != model.predict(x) for y, x in val_data])
    error_percentage = error_count / len(val_data) * 100

    if opts.test is None:
        print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))
    else:
        print("Test error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))


if __name__ == '__main__':
    main(sys.argv[1:])
