import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)
path = "../input/"

train = pd.read_csv(path+"train_porter.csv")
test = pd.read_csv(path+"test_porter.csv")
test['is_duplicated']=[-1]*test.shape[0]

len_train = train.shape[0]

data_all = pd.concat([train,test])


def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / len(a)

print('Generate intersection')
train_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
test_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
pd.to_pickle(train_interaction,path+"train_interaction.pkl")
pd.to_pickle(test_interaction,path+"test_interaction.pkl")

print('Generate porter intersection')
train_porter_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
test_porter_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)

pd.to_pickle(train_porter_interaction,path+"train_porter_interaction.pkl")
pd.to_pickle(test_porter_interaction,path+"test_porter_interaction.pkl")
