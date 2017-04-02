import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
seed=1024
np.random.seed(seed)
path = "../input/"
train = pd.read_csv(path+"train_porter.csv")


# tfidf
train_question1_tfidf = pd.read_pickle(path+'train_question1_tfidf.pkl')[:]
test_question1_tfidf = pd.read_pickle(path+'test_question1_tfidf.pkl')[:]

train_question2_tfidf = pd.read_pickle(path+'train_question2_tfidf.pkl')[:]
test_question2_tfidf = pd.read_pickle(path+'test_question2_tfidf.pkl')[:]


train_question1_porter_tfidf = pd.read_pickle(path+'train_question1_porter_tfidf.pkl')[:]
test_question1_porter_tfidf = pd.read_pickle(path+'test_question1_porter_tfidf.pkl')[:]

train_question2_porter_tfidf = pd.read_pickle(path+'train_question2_porter_tfidf.pkl')[:]
test_question2_porter_tfidf = pd.read_pickle(path+'test_question2_porter_tfidf.pkl')[:]


train_interaction = pd.read_pickle(path+'train_interaction.pkl')[:].reshape(-1,1)
test_interaction = pd.read_pickle(path+'test_interaction.pkl')[:].reshape(-1,1)

train_porter_interaction = pd.read_pickle(path+'train_porter_interaction.pkl')[:].reshape(-1,1)
test_porter_interaction = pd.read_pickle(path+'test_porter_interaction.pkl')[:].reshape(-1,1)


train_jaccard = pd.read_pickle(path+'train_jaccard.pkl')[:].reshape(-1,1)
test_jaccard = pd.read_pickle(path+'test_jaccard.pkl')[:].reshape(-1,1)

train_porter_jaccard = pd.read_pickle(path+'train_porter_jaccard.pkl')[:].reshape(-1,1)
test_porter_jaccard = pd.read_pickle(path+'test_porter_jaccard.pkl')[:].reshape(-1,1)

train_len = pd.read_pickle(path+"train_len.pkl")
test_len = pd.read_pickle(path+"test_len.pkl")
scaler = MinMaxScaler()
scaler.fit(np.vstack([train_len,test_len]))
train_len = scaler.transform(train_len)
test_len =scaler.transform(test_len)


X = ssp.hstack([
    train_question1_tfidf,
    train_question2_tfidf,
    train_interaction,
    train_porter_interaction,
    train_jaccard,
    train_porter_jaccard,
    train_len,
    ]).tocsr()


y = train['is_duplicate'].values[:]

X_t = ssp.hstack([
    test_question1_tfidf,
    test_question2_tfidf,
    test_interaction,
    test_porter_interaction,
    test_jaccard,
    test_porter_jaccard,
    test_len,
    ]).tocsr()


print X.shape
print X_t.shape

skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

dump_svmlight_file(X,y,path+"X_tfidf.svm")
del X
dump_svmlight_file(X_t,np.zeros(X_t.shape[0]),path+"X_t_tfidf.svm")
del X_t

def oversample(X_ot,y,p=0.165):
    pos_ot = X_ot[y==1]
    neg_ot = X_ot[y==0]
    #p = 0.165
    scale = ((pos_ot.shape[0]*1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = ssp.vstack([neg_ot, neg_ot]).tocsr()
        scale -=1
    neg_ot = ssp.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
    ot = ssp.vstack([pos_ot, neg_ot]).tocsr()
    y=np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]]=1.0
    print y.mean()
    return ot,y

X_train,y_train = oversample(X_train.tocsr(),y_train,p=0.165)
X_test,y_test = oversample(X_test.tocsr(),y_test,p=0.165)

X_train,y_train = shuffle(X_train,y_train,random_state=seed)

dump_svmlight_file(X_train,y_train,path+"X_train_tfidf.svm")
dump_svmlight_file(X_test,y_test,path+"X_test_tfidf.svm")

