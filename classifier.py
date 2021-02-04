import processing

class Sentiment:
    def __init__(self, name, probability, dictionary):
        self.name = name
        self.probability = probability
        self.dictionary = dictionary
        self.type = "Sentiment"
    
    def getPredictionReply(self):
        return 'This tweet is {}'.format(self.name)

class Topic:
    def __init__(self, name, probability, dictionary):
        self.name = name
        self.probability = probability
        self.dictionary = dictionary
        self.type = "Topic"

    def getPredictionReply(self):
        return 'This tweet is about {}'.format(self.name)

# Bayes Naive therom methods

def getTotalCount(listCat):
    totalCount = 0
    for i in listCat:
        totalCount += len(i)
    return totalCount

def getProbability(listOfData, totalCount):
    return float(len(listOfData)) / totalCount

def condProb(wordCount,totalWordCount, vocab_count):
    prob = (wordCount + 1)/(float(totalWordCount) + float(vocab_count))
    return prob

def countWord(word, bag):
    count = 0
    for i in bag:
        if word == i:
            count += 1
    return count

# Gets name of prediction of each line
def predict(words, outcomes):
    words = words.split()
    clean_words = []
    for i in words:
        clean_words.append(processing.cleanText(i, processing.redundantChar))
    
    # Iterate through the sentiments and generate probability vector
    for i in outcomes:
        for j in clean_words:
            if j in i.dictionary:
                i.probability = i.probability * i.dictionary[j]

    highest = 0.0
    highestIndex = 0
    index = 0
    probs = []
    for x in outcomes:
        probs.append(x.probability)
    
    for i in probs:
        if (i > highest):
            highest = i
            highestIndex = index
        index += 1
    
    return outcomes[highestIndex]
