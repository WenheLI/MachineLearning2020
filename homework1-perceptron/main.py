from typing import Dict, List, Tuple
import pickle
import numpy as np

def words(data: List[str], X: int) -> List[str]:
    wordsCounter: Dict[str, int] = {}
    for email in data:
        # exclude label
        words = set(map(lambda x: x.strip('\n'), email.split(" ")[1:]))
        for word in words:
            counter = wordsCounter.get(word, 0)
            wordsCounter[word] = counter + 1
    res = []
    for k,v in wordsCounter.items():
        if (v >= X):
            res.append(k)
    print(res.__len__())
    return res

def feature_vector(email: str, vocabulary: List[str]) -> List[int]:
    email = email.strip('\n')
    words = email.split(' ')[1:]
    wordsSet = set(words)
    return list(map(lambda x: 1 if x in wordsSet else 0, vocabulary ))

def dotProduct(w: List[int], x: List[int]) -> int:
    res = 0
    for i in range(len(w)):
        res += w[i] * x[i]
    return res

def vecPlus(x: List[int], y: List[int]) -> List[int]:
    ret = [0] * len(x)
    for idx in range(len(y)):
        ret[idx] =  x[idx] + y[idx]
    return ret

def scalarByVec(x: int, y: List[int]) -> List[int]:
    return list(map(lambda val: val * x, y))

def perceptron_error(w, data):
    labels = data[0]
    vecs = data[1]
    res = list(map(lambda vec: 1 if dotProduct(w, vec) > 0 else -1, vecs))
    errors = 0
    for idx in range(len(res)):
        if (res[idx] != labels[idx]):
            errors += 1
    return errors / labels.__len__()
        

def perceptron_train(data, vocabulary):
    labels = data[0]
    vecs = data[1]
    w = [0] * len(vecs[0])
    epochs = 1
    totalMistake = 0
    while True and epochs <= 30:
        currMistake = 0
        for idx in range(len(vecs)):
            pred = labels[idx] * dotProduct(w, vecs[idx])
            if pred > 0:
                w = w
            else:
                currMistake += 1
                w = vecPlus(scalarByVec(labels[idx], vecs[idx]), w)
        print(epochs, currMistake, totalMistake)
        totalMistake += currMistake
        epochs += 1
        if currMistake == 0:
            break

    return (w, totalMistake, epochs)


if __name__ == "__main__":
    trainVecs = []
    trainLabels = []

    validVecs = []
    validLabels = []

    testVecs = []
    vocabularyList = []
    with open("spam_data/train.txt", 'r') as f:
        emails = f.readlines()
        vocabularyList = words(emails, 26)
        trainLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
        trainVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
    with open("spam_data/validation.txt", "r") as f:
        emails = f.readlines()
        validLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
        validVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
    with open("spam_data/spam_test.txt", "r") as f:
        emails = f.readlines()
        testVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
    w, k, iterations = perceptron_train([trainLabels, trainVecs], vocabularyList)
    trainError = perceptron_error(w, [trainLabels, trainVecs])
    print(trainError)
    validationError = perceptron_error(w, [validLabels, validVecs])
    print(validationError)
    sortedW = np.argsort(w)
    largest12 = sortedW[-12:]
    smallest12 = sortedW[:12]

    print("most positive weights are:")
    for idx in largest12:
        print(vocabularyList[idx])

    print("most negative weights are:")
    for idx in smallest12:
        print(vocabularyList[idx])
    with open("weight.pkl", "wb") as f:
        pickle.dump(w, f)