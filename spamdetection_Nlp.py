# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:54:21 2023

@author: DELL
"""

#################spam detection
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
data.head()

data = data[["class", "message"]]

x = np.array(data["message"])
y = np.array(data["class"])
cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = MultinomialNB()
clf.fit(X_train,y_train)

sample = input('Enter a message:')
data = cv.transform([sample]).toarray()
print(clf.predict(data))



'''

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:54:21 2023

@author: DELL
"""

#################spam detection
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\aiml\datasets\spam_task\spam.csv", encoding= 'latin-1')
data.head()

data = data[["v1", "v2"]]

x = np.array(data["v2"])
y = np.array(data["v1"])
cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = MultinomialNB()
clf.fit(X_train,y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

sample = input('Enter a message:')
data = cv.transform([sample]).toarray()
print(clf.predict(data))
 '''