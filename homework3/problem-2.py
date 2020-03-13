"""
Put your NetID here.
"""

import matplotlib.pyplot as pyplot
import sys
import argparse
import numpy as np
import math
from collections import Counter

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

    with open(filename, "r") as f:
        lines = list(map(lambda line: line.strip().split(), f.readlines()))
        return list(map(lambda line: (True if line[0] == "1" else False, line[1:]), lines))


def split_train(original_train_data):
    """
    Split the original training set into two sets:
        * the training set, and
        * the validation set.
    """

    # Fill in your code here.

    split_idx = int(len(original_train_data) * .8)
    return (original_train_data[1000:], original_train_data[:1000])

def create_wordlist(original_train_data, threshold=26):
    """
    Create a word list from the original training set.
    Only get a word if it appears in at least $threshold emails.
    """

    # Fill in your code here.
    words_dict = {}
    
    for data in original_train_data:
        data = list(set(data[1]))
        for d in data:
            if d in words_dict.keys():
                words_dict[d] += 1
            else:
                words_dict[d] = 1
    ret = list(map(lambda word: word[0], filter(lambda x: x[1] >= threshold, words_dict.items())))
    # print(len(ret))
    return ret



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

        pos_counts = len(list(filter(lambda d: d[0], data)))
        total = len(data)
        neg_counts = total - pos_counts
        # print(total)
        # print(neg_counts)
        # print(pos_counts)
        return neg_counts, pos_counts


    @staticmethod
    def count_words(wordlist, data):
        """
        Count the number of times that each word appears in emails under a given label.
        Returns (a numpy array):
            * word_counts: a numpy array with shape (2, L), where L is the length of $wordlist,
                - word_counts[0, i] represents the number of times that word $wordlist[i] appears in non-spam (negative) emails, and
                - word_counts[1, i] represents the number of times that word $wordlist[i] appears in spam (positive) emails.
        """

        # Fill in your code here.
        word_counts = [[0] * len(wordlist) for i in range(2)]
        for d in data:
            idx = 1 if d[0] else 0
            d = set(d[1])
            # print(d)
            for word_idx in range(len(wordlist)):
                if wordlist[word_idx] in d:
                    word_counts[idx][word_idx] += 1
        # print(max(word_counts[0]))
        # print(max(word_counts[1]))
        return np.asarray(word_counts)

    @staticmethod
    def calculate_probability(label_counts, word_counts, use_map=False):
        """
        Calculate the probabilities, both the prior and likelihood.
        Returns (a pair of numpy array):
            * prior_probs: a numpy array with shape (2, ), only two elements, where
                - prior_probs[0] is the prior probability of negative labels, and
                - prior_probs[1] is the prior probability of positive labels.
            * likelihood_probs: a numpy array with shape (2, L), where L is the length of the word list,
                - likelihood_probs[0, i] represents the likelihood probability of the $i-th word in the word list, given that the email is non-spam (negative), and
                - likelihood_probs[1, i] represents the likelihood probability of the $i-th word in the word list, given that the email is spam (positive).
        """

        # Fill in your code here.
        # Do not forget to add the additional counts.
        neg, pos = label_counts
        total = neg + pos
        prior_probs = np.asarray([neg/total, pos/total])
        word_counts_sum = word_counts.sum(axis=1)
        if use_map:
            likelihood_probs = list(map(lambda x, idx: (x+1)/(word_counts_sum[idx] + len(word_counts[idx])), word_counts, range(len(word_counts))))
        else:
            likelihood_probs = list(map(lambda x, idx: x/word_counts_sum[idx], word_counts, range(len(word_counts))))
        return prior_probs, np.asarray(likelihood_probs)

    def __init__(self, wordlist):
        self.wordlist = wordlist

    def fit(self, data, cal_error=False, use_map=False):
        label_counts = self.__class__.count_labels(data)
        word_counts = self.__class__.count_words(self.wordlist, data)
        
        self.use_map = use_map
        self.prior_probs, self.likelihood_probs = self.__class__.calculate_probability(label_counts, word_counts, use_map=use_map)
        if cal_error:
            error_count = sum([y != self.predict(x) for y, x in data])
            error_percentage = error_count / len(data) * 100
            return error_percentage
        # You may do some additional processing of variables here, if you want.
        # Suggestion: You may get the log of probabilities.

    def predict(self, x):
        """
        Predict whether email $x is a spam or not.
        Returns:
            * y: a boolean value indicating whether $x is a spam or not.
        """

        # Fill in your code here.
        neg = 0
        pos = 0

        c = Counter(x)
        if self.use_map:
            for idx in range(len(self.wordlist)):
                v = c.get(self.wordlist[idx], 0)
                for time in range(v):
                    neg += math.log(self.likelihood_probs[0][idx])
                    pos += math.log(self.likelihood_probs[1][idx])
                if v == 0:
                    neg += math.log(1 - self.likelihood_probs[0][idx])
                    pos += math.log(1 - self.likelihood_probs[1][idx])
            neg += math.log(self.prior_probs[0])
            pos += math.log(self.prior_probs[1])

            if pos - neg >= 0:
                return True
                
            return False
        else:
            neg = 1
            pos = 1
            for idx in range(len(self.wordlist)):
                v = c.get(self.wordlist[idx], 0)
                for time in range(v):
                    neg *= (self.likelihood_probs[0][idx])
                    pos *= (self.likelihood_probs[1][idx])
                if v == 0:
                    neg *= (1 - self.likelihood_probs[0][idx])
                    pos *= (1 - self.likelihood_probs[1][idx])
            neg *= (self.prior_probs[0])
            pos *= (self.prior_probs[1])

            if pos / neg >= 1:
                return True
                
            return False




def a(argv):
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

def b(argv):
    opts = Opts(argv)

    if opts.test is None:
        original_train_data = read_data(opts.train)
        train_data, val_data = split_train(original_train_data)
    else:
        original_train_data = read_data(opts.train)
        train_data = original_train_data
        val_data = read_data(opts.test)
    sizeN = [200, 400, 800, 1600, 2400, 3200, 4000]
    validation = []
    train = []
    for size in sizeN:
        # Create the word list.
        wordlist = create_wordlist(original_train_data, opts.threshold)

        model = Model(wordlist)
        train.append(model.fit(train_data[:size], cal_error=True, use_map=True))

        error_count = sum([y != model.predict(x) for y, x in val_data])
        error_percentage = error_count / len(val_data) * 100
        print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))
        validation.append(error_percentage)
    trainLine = pyplot.plot(sizeN, train, label="train")
    validationLine = pyplot.plot(sizeN, validation, label="validation")
    pyplot.title("Training Size V.S. Error Rate")
    pyplot.xlabel("Training Size")
    pyplot.ylabel("Error Rate")
    pyplot.legend(["train", "validation"])
    pyplot.show()

def c(argv):
    opts = Opts(argv)

    original_train_data = read_data(opts.train)
    train_data, val_data = split_train(original_train_data)

    config = [i for i in range(19, 29)]
    validation = []
    for size in config:
        # Create the word list.
        wordlist = create_wordlist(original_train_data, size)

        model = Model(wordlist)
        model.fit(train_data, use_map=True)

        error_count = sum([y != model.predict(x) for y, x in val_data])
        error_percentage = error_count / len(val_data) * 100
        print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))
        validation.append(error_percentage)
    pyplot.plot(config, validation, "r")
    pyplot.title("Threshold V.S. Error Rate")
    pyplot.xlabel("Threshold")
    pyplot.ylabel("Error Rate")
    pyplot.show()

def d(argv):
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

    print("MLE::Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))

    model = Model(wordlist)
    model.fit(train_data, use_map=True)

    error_count = sum([y != model.predict(x) for y, x in val_data])
    error_percentage = error_count / len(val_data) * 100

    print("MAP::Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))

def main(argv):
    # a(argv)
    # b(argv)
    # c(argv)
    d(argv)

if __name__ == '__main__':
    main(sys.argv[1:])

