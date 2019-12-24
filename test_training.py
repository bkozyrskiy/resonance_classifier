
from utils import  read_params
import os
import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'data_loader'))
from data import DataBuildClassifier

from train_clf import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import keras.backend as K
from keras.utils import to_categorical
import codecs
import numpy as np

def get_data():
    data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    all_subjects = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]
    params = read_params('config.json')
    subjects = data.get_data(all_subjects, shuffle=False, windows=[(0.2, 0.5)], baseline_window=(0.2, 0.3),
                             resample_to=params['resampled_to'])
    return subjects

def naive_evaluate(model,test_data):
    x_tst,y_tst = test_data
    x_tst = x_tst[:,np.newaxis,:,:]
    predictions = model.predict(x_tst)[:,1]
    test_auc_naive = roc_auc_score(y_tst[:,1], predictions)
    return test_auc_naive


if __name__ == '__main__':
    all_subjects = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]
    subjects = get_data()
    val_aucs = []
    test_aucs_naive = []
    with codecs.open('res.txt','w', encoding='utf8') as f:
        f.write('subj,mean_val_aucs,test_aucs_naive\n')
    for subject in subjects:
        x = subjects[subject][0]
        x = x.transpose([0,2,1])
        y = subjects[subject][1]
        x_tr_val_ind, x_tst_ind, y_tr_val, y_tst = train_test_split(range(x.shape[0]), y, test_size=0.2, stratify=y)
        x_tr_val = x[x_tr_val_ind,...]
        x_tst = x[x_tst_ind,...]
        trained_model,val_auc = train_model(x_tr_val, y_tr_val)
        test_auc = naive_evaluate(trained_model,(x_tst,to_categorical(y_tst,2)))
        K.clear_session()
        val_aucs.append(val_auc)
        test_aucs_naive.append(test_auc)

        with codecs.open('res.txt','a', encoding='utf8') as f:
            f.write(u'%s, %.02f, %.02f\n' %(subject,val_auc,test_auc))
    with codecs.open('res.txt', 'a', encoding='utf8') as f:
        f.write(u'MEAN, %.02f±%.02f, %.02f%.02f\n' % (np.mean(val_aucs),np.std(val_aucs), np.mean(test_aucs_naive),np.std(test_aucs_naive)))





