from model import EEGNet
from sklearn.model_selection import StratifiedKFold
import os
from utils import  AucMetricHistory, single_auc_loging,read_params
import numpy as np
import time
from keras.utils import to_categorical


def cv_per_subj_test(x_tr_val,y_tr_val,model,model_path,max_epochs, block_mode = False,plot_fold_history=True):
    '''

    :param x_tr_val:
    :param y_tr_val:
    :param model:
    :param model_path:
    :param max_epochs:
    :param block_mode:
    :param plot_fold_history:
    :return:
    '''
    model.save_weights('tmp.h5') # Nasty hack. This weights will be used to reset model
    same_subj_auc = AucMetricHistory()

    best_val_auc_epochs = []
    best_val_aucs = []
    best_val_loss_epochs = []

    folds = 4  # To preserve split as 0.6 0.2 0.2
    # if block_mode:
        # targ_indices = [ind for ind in range(len(y)) if y[ind,1] == 1]
        # nontarg_indices = [ind for ind in range(len(y)) if y[ind,1] == 0]
        # tst_ind = targ_indices[int(0.8*len(targ_indices)):] + nontarg_indices[int(0.8*len(nontarg_indices)):]
        #
        #
        #
        #
        # x_tr, x_tst = x[targ_indices[:int(0.8*len(targ_indices))] + nontarg_indices[:int(0.8*len(nontarg_indices))]],x[tst_ind]
        # y_tr, y_tst = y[targ_indices[:int(0.8*len(targ_indices))] + nontarg_indices[:int(0.8*len(nontarg_indices))]],y[tst_ind]
        #
        # targ_tr_ind = list(range(int(0.8*len(targ_indices))))
        # nontarg_tr_ind = list(range(int(0.8*len(targ_indices)),int(0.8*len(targ_indices)) + int(0.8*len(nontarg_indices))))
        #
        #
        # targ_sections = list(map(int,np.linspace(0,1,folds+1)*len(targ_tr_ind)))
        # nontarg_sections = list(map(int, np.linspace(0, 1, folds + 1) * len(nontarg_tr_ind)))
        #
        # cv_splits=[]
        # for ind in range(folds):
        #
        #     cv_splits.append(
        #         (targ_tr_ind[targ_sections[0]:targ_sections[ind]] + targ_tr_ind[targ_sections[ind+1]:targ_sections[-1]] + \
        #         nontarg_tr_ind[nontarg_sections[0]:nontarg_sections[ind]] + nontarg_tr_ind[nontarg_sections[ind + 1]:nontarg_sections[-1]],
        #         targ_tr_ind[targ_sections[ind]:targ_sections[ind + 1]] + nontarg_tr_ind[nontarg_sections[ind]:nontarg_sections[ind + 1]])
        #     )

    # else:
    cv = StratifiedKFold(n_splits=folds,shuffle=True)
    cv_splits = list(cv.split(x_tr_val, y_tr_val[:,1]))

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
    # for fold, (train_idx, val_idx) in enumerate(cv.split(x_tr, y_tr)):
        fold_model_path = os.path.join(model_path,'%d' % fold)
        os.makedirs(fold_model_path)
        # make_checkpoint = ModelCheckpoint(os.path.join(fold_model_path, '{epoch:02d}.hdf5'),
        #                                   monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
        model.load_weights('tmp.h5') # Rest model on each fold
        x_tr_fold,y_tr_fold = x_tr_val[train_idx],y_tr_val[train_idx]
        x_val_fold, y_val_fold = x_tr_val[val_idx], y_tr_val[val_idx]
        val_history = model.fit(x_tr_fold, y_tr_fold, epochs = max_epochs , validation_data=(x_val_fold, y_val_fold),
                            callbacks=[same_subj_auc], batch_size=64, shuffle=True)

        best_val_loss_epochs.append(np.argmax(val_history.history['val_loss']) + 1)
        best_val_auc_epochs.append(np.argmax(val_history.history['val_auc']) + 1) # epochs count from 1 (not from 0)
        best_val_aucs.append(np.max(val_history.history['val_auc']))
        if plot_fold_history:
            single_auc_loging(val_history.history, 'fold %d' % fold, fold_model_path)


    #Test  performance (Naive, until best epoch
    model.load_weights('tmp.h5') # Rest model before traning on train+val

    model.fit(x_tr_val, y_tr_val, epochs=int(np.mean(best_val_loss_epochs)),batch_size=64, shuffle=True)
    model.save(os.path.join(model_path,'final_%d.hdf5' %int(np.mean(best_val_loss_epochs))))
    os.remove('tmp.h5')
    return model, np.mean(best_val_aucs)


# class ResonanceClassifier(object):
#     def __init__(self,path_to_config,num_channels,num_samples):
#         '''
#
#         :param path_to_config: path to json file with parameters
#         '''
#         parameters = read_params(path_to_config)
#         self.network_hp = parameters['classifier_hyperparameters']
#         self.resampled_to = parameters['resample_to']
#         self.model = self._create_model()
#
#
#
#     def fit(self,x,y):
#         '''
#         :param x: EEG [trials x channels x samples]
#
#         '''
#     def predict(self,x):

def create_model(params, num_channels, num_samples):
    hyperparams = params['classifier_hyperparameters']
    hyperparams['F2'] = hyperparams['F1'] * hyperparams['D']
    model = EEGNet(hyperparams, nb_classes=2, Chans=num_channels, Samples=num_samples, kernLength=64,
                   dropoutType='SpatialDropout2D')
    return model

def train_model(epochs, labels):
    '''

    :param epochs:  eeg epochs of shape trials x channels x samples
    :return:
    '''
    params = read_params('config.json')
    model  = create_model(params=params, num_channels=epochs.shape[1],num_samples=epochs.shape[2])

    path_to_models_dir = params['path_to_models_dir']
    path_to_model = os.path.join(path_to_models_dir,str(int(time.time())))
    max_epochs = params['max_epochs']
    labels = to_categorical(labels,2)
    epochs = epochs[:,np.newaxis,:,:]
    model, mean_val_aucs = cv_per_subj_test(x_tr_val=epochs,
                                            y_tr_val=labels,
                                            model=model,
                                            model_path=path_to_model,
                                            max_epochs=max_epochs, block_mode=False, plot_fold_history=True)
    return model,mean_val_aucs


