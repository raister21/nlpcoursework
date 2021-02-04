import classifier, processing

# Initializing training data
positive = ["i love going to the movies", "I love the weather", "I like summer times", "i would love to see that movie","i like programming, it is my passion", "It's going to be good"]
neutral = ["what is the meaning of life", "i just can't wait for 2022", "i will be going for football tomorrow", "It's okay, it's not that bad", "what a boring day"]
negative = ["captian marvel was a bad movie", "i hate the weather today, it's too cold", "i hate to play baskeball", "i hate swimming as well","school sucks, i hate it", "Fuck this, i hate it"]

personal = ["i'm not feeling it today", "what would she say?", "should i invest in stock?", "i will not be entertaining lazy people","what is the meaning of life"]
weather = ["I love the weather", "i hate the weather today, it's too cold", "hope it does not rain tomorrow", "when will summer come!"]
sports = ["i will be going for football tomorrow","i dont like to play baskeball","i dont like swimming as well", "table tenis is a stupid sports!"]
politics = ["i hope trump wins the elections", "bernie sanders is a funny guy","wow i cant believe that joe biden won the election", "please vote if you can, it is for the good of the country"]

sentimentDataList = [positive, neutral, negative]
topicDataList = [personal, weather, sports, politics]

# Declaring variables for sentiment
sentimentsWordBag = []
positive_bag = []
neutral_bag = []
negative_bag = []

positiveDict = {}
neutralDict = {}
negativeDict = {}

# Declaring variables for topic
topicsWordBag = []
personal_bag = []
weather_bag = []
sports_bag = []
politics_bag = []

personalDict = {}
weatherDict = {}
sportsDict = {}
politicsDict = {}

# generating vectors
processing.vectorize(positive, sentimentsWordBag, positive_bag)
processing.vectorize(neutral, sentimentsWordBag, neutral_bag)
processing.vectorize(negative, sentimentsWordBag, negative_bag)

processing.vectorize(personal, topicsWordBag, personal_bag)
processing.vectorize(weather, topicsWordBag, weather_bag)
processing.vectorize(sports, topicsWordBag, sports_bag)
processing.vectorize(politics, topicsWordBag, politics_bag)

# Generating vocabulary and data for bayes naive theorem
sentimentsWordBag = set(sentimentsWordBag)
total_sentiment_count = classifier.getTotalCount(sentimentDataList)
positive_probability = classifier.getProbability(positive, total_sentiment_count)
neutral_probability = classifier.getProbability(neutral, total_sentiment_count)
negative_probability = classifier.getProbability(negative, total_sentiment_count)

topicsWordBag = set(topicsWordBag)
total_topics_count = classifier.getTotalCount(topicDataList)
personal_probability = classifier.getProbability(personal, total_topics_count)
weather_probability = classifier.getProbability(weather, total_topics_count)
sports_probability = classifier.getProbability(sports, total_topics_count)
politics_probability = classifier.getProbability(politics, total_topics_count)


# Making dictionary
processing.makeDictionary(sentimentsWordBag, positiveDict, positive_bag)
processing.makeDictionary(sentimentsWordBag, neutralDict, neutral_bag)
processing.makeDictionary(sentimentsWordBag, negativeDict, negative_bag)

processing.makeDictionary(topicsWordBag, personalDict, personal_bag)
processing.makeDictionary(topicsWordBag, weatherDict, weather_bag)
processing.makeDictionary(topicsWordBag, sportsDict, sports_bag)
processing.makeDictionary(topicsWordBag, politicsDict, politics_bag)

# Creating sentiments
positiveSentiment = classifier.Sentiment("Positive", positive_probability, positiveDict)
neutralSentiment = classifier.Sentiment("Neutral", neutral_probability, neutralDict)
negativeSentiment = classifier.Sentiment("Negative", negative_probability, negativeDict)

# Creating topics
personaltopic = classifier.Topic("Personal", personal_probability, personalDict)
weathertopic = classifier.Topic("Weather",weather_probability, weatherDict)
sportstopic = classifier.Topic("Sports", sports_probability, sportsDict)
politicstopic = classifier.Topic("Politics", politics_probability, politicsDict)

sentiments = [positiveSentiment, neutralSentiment, negativeSentiment]
topics = [personaltopic, weathertopic, sportstopic, politicstopic]

# print(classifier.predict("i hate that trump won the election", sentiments))
# print(classifier.predict("i hate that trump won the elections", topics))

tweet = input("What would you like to tweet?")

sentimentPrediction = classifier.predict(tweet, sentiments)
topicPrediction = classifier.predict(tweet, topics)

print("Your tweet about {}, is {}".format(topicPrediction.name, sentimentPrediction.name))
