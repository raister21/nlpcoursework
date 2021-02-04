import classifier

redundantChar = [',','.','!','?']


def cleanText(text,listOfChar):
    for i in listOfChar:
        text.strip(i)
    text.lower()
    return text

def vectorize(dataList, wordBag, bag):
    for i in dataList:
        splitted = i.split()
        for j in splitted: 
            wordBag.append(cleanText(j,redundantChar))
            bag.append(cleanText(j,redundantChar))

def makeDictionary(wordbag, dictionary, bag):
    bagWordCount = len(bag)
    vocabCount = len(wordbag)
    # Mapping dictionary
    for i in wordbag:
        dictionary[i] = 0
    
    for x in dictionary.keys():
        counted = classifier.countWord(x, bag)
        dictionary[x] = classifier.condProb(counted, bagWordCount, vocabCount)
