import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)
path = "../input/"

ft = ['question1','question2','question1_porter','question2_porter']
train = pd.read_csv(path+"train_porter.csv")[ft]
test = pd.read_csv(path+"test_porter.csv")[ft]
# test['is_duplicated']=[-1]*test.shape[0]

len_train = train.shape[0]

data_all = pd.concat([train,test])
print data_all

max_features = None
ngram_range = (1,2)
min_df = 3
print('Generate tfidf')
feats= ['question1','question2']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus+=data_all[f].values.tolist()

vect_orig.fit(
    corpus
    )

for f in feats:
    tfidfs = vect_orig.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:train.shape[0]]
    test_tfidf = tfidfs[train.shape[0]:]
    pd.to_pickle(train_tfidf,path+'train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf,path+'test_%s_tfidf.pkl'%f)


print('Generate porter tfidf')
feats= ['question1_porter','question2_porter']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus+=data_all[f].values.tolist()

vect_orig.fit(
    corpus
    )

for f in feats:
    tfidfs = vect_orig.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:train.shape[0]]
    test_tfidf = tfidfs[train.shape[0]:]
    pd.to_pickle(train_tfidf,path+'train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf,path+'test_%s_tfidf.pkl'%f)

