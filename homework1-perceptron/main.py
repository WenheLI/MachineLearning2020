from typing import Dict, List, Tuple
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    l = list(map(lambda x: 1 if x in wordsSet else 0, vocabulary ))
    l.insert(0, 1)
    return l

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
    w = [0] * (len(vecs[0]) )
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

def buildTrainVecs(X, file):
    trainVecs = []
    trainLabels = []
    with open(file, 'r') as f:
        emails = f.readlines()
        vocabularyList = words(emails, X)
        trainLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
        trainVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
        print(trainVecs[0].__len__())
        return (trainVecs, trainLabels, vocabularyList)

def vecL2Norm(vec):
    length = 0
    for x in vec:
        length += x ** 2
    return length

def q1_3():
    validVecs = []
    validLabels = []  
    trainVecs, trainLabels, vocabularyList = buildTrainVecs(26, "spam_data/train.txt")
    with open("spam_data/validation.txt", "r") as f:
        emails = f.readlines()
        validLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
        validVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
    w, k, iterations = perceptron_train([trainLabels, trainVecs], vocabularyList)
    print("Total mistakes are: " + str(k) + ". With iterations:" + str(iterations))
    trainError = perceptron_error(w, [trainLabels, trainVecs])
    print(trainError)
    validationError = perceptron_error(w, [validLabels, validVecs])
    print(validationError)
    with open("./weight.pkl", "wb") as f:
        pickle.dump(w, f)
    with open("./voc.pkl", "wb") as f:
        pickle.dump(vocabularyList, f)

def q1_4():
    v = []
    with open("./voc.pkl", "rb") as f:
        v = pickle.load(f)
    with open("./weight.pkl", "rb") as f:
        w = pickle.load(f)
        
        # exclude bias w0
        w = w[1:]
        print(len(w))
        sortedW = np.argsort(w)
        largest12 = sortedW[-12:]
        smallest12 = sortedW[:12]
        largest12 = reversed(largest12)
        print("most positive weights are:")
        for large in largest12:
            print(v[large], end="\n")
        print()
        print("most negative weights are:")
        for small in smallest12:
            print(v[small], end="\n")

def q1_5and6():
    validVecs = []
    validLabels = []  
    trainVecs, trainLabels, vocabularyList = buildTrainVecs(26, "spam_data/train.txt")
    with open("spam_data/validation.txt", "r") as f:
        emails = f.readlines()
        validLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
        validVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
    validationErrors = []
    iters = []
    exps = [200, 600, 1200, 2400, 4000]
    for rowNum in exps:
        w, _, i = perceptron_train([trainLabels[:rowNum], trainVecs[:rowNum]], vocabularyList)
        e = perceptron_error(w, [validLabels, validVecs])
        validationErrors.append(e)
        iters.append(i)
    plt.plot(exps, validationErrors)
    plt.title("Validation error with different size of training data")
    plt.xlabel("training set")
    plt.ylabel("validation error")
    plt.show()
    plt.plot(exps, iters)
    plt.title("Iteractions with different size of training data")
    plt.xlabel("training set")
    plt.ylabel("Iteractions")
    plt.show()
        
def q1_8():
    config = [i for i in range(22, 29)]
    validationErrors = []
    for c in config:
        trainVecs, trainLabels, vocabularyList = buildTrainVecs(c, "spam_data/train.txt")
        with open("spam_data/validation.txt", "r") as f:
            emails = f.readlines()
            validLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
            validVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
        w, _, _ = perceptron_train([trainLabels, trainVecs], vocabularyList)
        e = perceptron_error(w, [validLabels, validVecs])
        validationErrors.append(e)
    plt.plot(config, validationErrors)
    plt.title("Validation error with different X value")
    plt.xlabel("training set")
    plt.ylabel("validation error")
    plt.show()


    trainVecs, trainLabels, vocabularyList = buildTrainVecs(22, "spam_data/spam_train.txt")
    with open("spam_data/spam_test.txt", "r") as f:
        emails = f.readlines()
        testLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
        testVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
    w, _, _ = perceptron_train([trainLabels, trainVecs], vocabularyList)
    e = perceptron_error(w, [testLabels, testVecs])
    print(e)

def q1_9():
    trainVecs, trainLabels, vocabularyList = buildTrainVecs(1500, "spam_data/train.txt")
    with open("spam_data/validation.txt", "r") as f:
        emails = f.readlines()
        validLabels = list(map(lambda x: 1 if int(x.split(' ')[0]) == 1 else -1, emails))
        validVecs = list(map(lambda x: feature_vector(x, vocabularyList), emails))
    w, _, _ = perceptron_train([trainLabels, trainVecs], vocabularyList)
    e = perceptron_error(w, [validLabels, validVecs])
    print(e)

def q4():
    trainVecs, trainLabels, vocabularyList = buildTrainVecs(26, "spam_data/train.txt")
    maxNorm = max(list(map(lambda x: vecL2Norm(x), trainVecs)))
    print(maxNorm)
if __name__ == "__main__":
    # q1_3()
    # q1_4()
    # q1_5and6()
    # q1_8()
    # q1_9()
    q4()
