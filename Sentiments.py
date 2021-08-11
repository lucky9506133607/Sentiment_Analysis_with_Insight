import pandas as pd

df =  pd.read_csv('balanced_reviews.csv')

df.isnull().any(axis = 0)

#handle the missing data
df.dropna(inplace =  True)

#leaving the reviews with rating 3 and collect reviews with
#rating 1, 2, 4 and 5 onyl

df = df [df['overall'] != 3]

import numpy as np

#creating a label
#based on the values in overall column
df['Positivity'] = np.where(df['overall'] > 3 , 1 , 0)

#NLP
#reviewText - feature - df['reviewText']
#Positivity - label - df['Positivity']

import nltk
import re
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer #find root of any word and append it into corpus list

from nltk.corpus import stopwords

corpus = []


"""
for i in range(0,df.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ' , df.iloc[i,1])
    review = review.lower()
    
    review = review.split()
    
    #remove the stopwords
    review = [word for word in review if not word in stopwords.words('english')]
    
    #stemming
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review]
    
    review = " ".join(review)
    corpus.append(review)
"""

"""         here we can create new file for corpus data
import pandas as pd
df1= pd.DataFrame(data={"col1":corpus})    
df1.to_csv('corpus_data.csv')    
""" 

corpus_df = pd.read_csv('corpusnew_data.csv')
corpus_df.isnull().any(axis=0)
df_corpus_col = corpus_df.iloc[:,-1]
for i in range(0,len(df_corpus_col)):
    corpus.append(df_corpus_col[i])
    
"""
#features - corpus
#label -  df.iloc[:,-1]
    
from sklearn.feature_extraction.text import CountVectorizer

features_vectorized = CountVectorizer().fit_transform(corpus)

#fit - prepare the vocabulary
#transform - create so many columns, fill that matrix

labels = df.iloc[:,-1]
"""

#version 02
#tf-idf 
#term frequency inverse document frequency

"""
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 )



from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df = 5).fit(features_train)


features_train_vectorized = vect.transform(features_train)

#model building

from sklearn.linear_model import LogisticRegression


model = LogisticRegression()

model.fit(features_train_vectorized, labels_train)

predictions = model.predict(vect.transform(features_test))


from sklearn.metrics import confusion_matrix

confusion_matrix(labels_test, predictions)

from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)


#save - pickle format

#this below code will run on my machine 

import pickle

file  = open("pickle_model.pkl","wb")
pickle.dump(model, file)


#pickle the vocabulary
pickle.dump(vect.vocabulary_, open('features.pkl', 'wb'))

"""


#extended version -2
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(corpus)
features_train_vectorized = vect.transform(corpus)


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features_train_vectorized, df.iloc[0:518918,-1], random_state = 42 )


#model building

from sklearn.linear_model import LogisticRegression


model = LogisticRegression()

model.fit(features_train, labels_train)

predictions = model.predict(features_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(labels_test, predictions)

from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)


#save - pickle format

#this below code will run on my machine 

import pickle

file  = open("pickle_model.pkl","wb")
pickle.dump(model, file)


#pickle the vocabulary
pickle.dump(vect.vocabulary_, open('features.pkl', 'wb'))


