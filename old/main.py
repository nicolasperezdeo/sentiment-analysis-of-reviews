# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np



def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, features="html5lib").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)

train = pd.read_csv("all/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)


# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if (i+1) % 1000 == 0:
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(clean_train_reviews, np.asarray(train["sentiment"]), test_size=0.2)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)


train_data_features = X_train
train_data_labels = y_train
test_data_features = X_test
test_data_labels = y_test

train_data_features = vectorizer.fit_transform(X_train)
train_data_features = train_data_features.toarray()


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(X_test)
test_data_features = test_data_features.toarray()


# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, train_data_labels)

prediction = forest.predict(test_data_features)


# Compute the final score for bag of words
score = (np.sum((prediction == test_data_labels))/5000)*100
print(score)




