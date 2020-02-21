def writeFile(fileName: str, data: [str]):
    with open(fileName, "w") as f:
        f.writelines(data)

def dataSplitter(fileName: str):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        validation = lines[:1000]
        train = lines[1000:]
        writeFile("spam_data/train.txt", train)
        writeFile("spam_data/validation.txt", validation)
        print(len(validation))
        print(len(train))
        

if __name__ == "__main__":
    dataSplitter("spam_data/spam_train.txt")
