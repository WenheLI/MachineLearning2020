"""
Put your NetID here.
"""


import sys
import argparse
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as pyplot

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

    split_end = int(0.8 * len(original_train_data))
    return original_train_data[1000:], original_train_data[:1000]


def create_wordlist(original_train_data, threshold=26):
    """
    Create a word list from the original training set.
    Only get a word if it appears in at least $threshold emails.
    """

    # Fill in your code here.
    words_dict = {}
    stopwords = []
    with open("./stopwords.txt", "r") as f:
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    
    for data in original_train_data:
        data = list(set(data[1]))
        for d in data:
            if d not in set(stopwords):
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
        neg_counts, pos_counts = label_counts
        wordlist_length = len(wordlist)

        neg_emails = list(filter(lambda x: not x[0], data))
        pos_emails = list(filter(lambda x: x[0], data))

        negative_word_counts = np.zeros((wordlist_length, neg_counts))
        positive_word_counts = np.zeros((wordlist_length, pos_counts))


        for j in range(len(neg_emails)):
            d = Counter(neg_emails[j][1])
            for i in range(wordlist_length):
                val = d.get(wordlist[i], 0)
                negative_word_counts[i][j] += val

        for j in range(len(pos_emails)):
            d = Counter(pos_emails[j][1])
            for i in range(wordlist_length):
                val = d.get(wordlist[i], 0)
                positive_word_counts[i][j] += val

        return [negative_word_counts, positive_word_counts]

        



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

        neg_counts, pos_counts = label_counts
        total = pos_counts + neg_counts
        prior_probs = np.asarray([neg_counts/total, pos_counts/total])

        wordlist_length, _ = negative_word_counts.shape
        likelihood_mus =  np.zeros((2, wordlist_length))

        for idx in range(len(negative_word_counts)):
            likelihood_mus[0][idx] += (sum(negative_word_counts[idx])+1) / (negative_word_counts.sum()+wordlist_length)
        for idx in range(len(positive_word_counts)):
            likelihood_mus[1][idx] += (sum(positive_word_counts[idx])+1) / (positive_word_counts.sum()+wordlist_length)

        likelihood_sigmas = np.ones((2, wordlist_length))
        likelihood_sigmas *= 0.00001
        for idx in range(len(negative_word_counts)):
            likelihood_sigmas[0][idx] += sum(map(lambda x: (x - likelihood_mus[0][idx])**2, negative_word_counts[idx])) / (negative_word_counts.sum())
        for idx in range(len(positive_word_counts)):
            likelihood_sigmas[1][idx] += sum(map(lambda x: (x - likelihood_mus[1][idx])**2, positive_word_counts[idx])) /  (positive_word_counts.sum())

        return prior_probs, likelihood_mus, likelihood_sigmas

    def __init__(self, wordlist):
        self.wordlist = wordlist

    def fit(self, data, cal_error=False):
        label_counts = self.__class__.count_labels(data)
        negative_word_counts, positive_word_counts = self.__class__.count_words(self.wordlist, label_counts, data)
        self.prior_probs, self.likelihood_mus, self.likelihood_sigmas = self.__class__.calculate_probability(label_counts, negative_word_counts, positive_word_counts)
        if cal_error:
            error_count = sum([y != self.predict(x) for y, x in data])
            error_percentage = error_count / len(data) * 100
            return error_percentage
        # You may do some additional processing of variables here, if you want.

    def predict(self, x):
        """
        Predict whether email $x is a spam or not.
        Returns:
            * y: a boolean value indicating whether $x is a spam or not.
        """

        # Fill in your code here.

        def norm_func(x, miu, theta):
            base = -.5*math.log(2*math.pi*theta)
            power = -.5 * ((x - miu) ** 2)/theta
            return base + power

        neg = 1
        pos = 1

        counters = Counter(x)
        for idx in range(len(self.wordlist)):
            val = counters.get(self.wordlist[idx], 0)
    
            neg += norm_func(val, self.likelihood_mus[0][idx], self.likelihood_sigmas[0][idx])
            pos += norm_func(val, self.likelihood_mus[1][idx], self.likelihood_sigmas[1][idx])
        neg += math.log(self.prior_probs[0])
        pos += math.log(self.prior_probs[1])

        if neg - pos >= 0:
            return False
        return True


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
        train.append(model.fit(train_data[:size], cal_error=True))

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
        model.fit(train_data)

        error_count = sum([y != model.predict(x) for y, x in val_data])
        error_percentage = error_count / len(val_data) * 100
        print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))
        validation.append(error_percentage)
    pyplot.plot(config, validation, "r")
    pyplot.title("Threshold V.S. Error Rate")
    pyplot.xlabel("Threshold")
    pyplot.ylabel("Error Rate")
    pyplot.show()

def main(argv):
    #a(argv)
    b(argv)
    c(argv)
    



if __name__ == '__main__':
    main(sys.argv[1:])
