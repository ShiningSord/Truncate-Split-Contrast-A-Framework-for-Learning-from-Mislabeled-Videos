import os
import copy
from types import new_class
import time

import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA    
import multiprocessing as mp
from functools import partial
import torch 
#from pprint import pprint

def innitail_split(feature, gtlabel):
    reallyclean = gtlabel[:,0] == gtlabel[:,1]
    label = gtlabel[:,1]
    classnum = int(np.max(label)+1)
    isclean = np.zeros(feature.shape[0]).astype(bool)
    for i in range(classnum):
        predict = KMeans(n_clusters=2, n_init=10).fit(feature[label==i]).labels_
        if np.sum(predict==0) > np.sum(predict==1):
            isclean[label==i] =(predict==0)
        else:
            isclean[label==i] =(predict==1)
        print('class {} clean {} noisy {}'.format(i, isclean[label==i].sum(), (label==i).sum() - isclean[label==i].sum()))
    print('Ture {} False {}'.format((reallyclean==isclean).sum(), (reallyclean!=isclean).sum()))
    return isclean


def select_feature(clean_set, k = 30, K = 60):
    '''
    Input: C * N * L 3D clean_set
    Output: C * K regular used feature number
    '''
    
    C, N, L = clean_set.shape
    res = np.zeros([C, K])
    cnt = np.zeros([C, L])

    index = np.argsort(clean_set, axis=2)[:,:,::-1][:,:,:k]
    for i in range(C):
        for j in range(N):
            cnt[i][index[i][j]] += 1
        
    res = np.argsort(cnt, axis=1)[:,::-1][:,:K]
    return res
def select_feature_no_clean(AllFeature, label, k = 30, K = 60):
    '''
    Input: C * N * L 3D clean_set
    Output: C * K regular used feature number
    '''
    C = int(np.max(label) + 1)
    N, L = AllFeature.shape
    res = np.zeros([C, K])
    cnt = np.zeros([C, L])

    index = np.argsort(AllFeature, axis=1)[:,::-1][:,:k]
    for i in range(N):
        cnt[int(label[i])][index[i]] += 1
        
    res = np.argsort(cnt, axis=1)[:,::-1][:,:K]
    return res

def select_feature_var_no_clean(AllFeature, label, k = 30, K = 60,start=0):
    '''
    Input: C * N * L 3D clean_set
    Output: C * K regular used feature number
    '''
    C = int(np.max(label) + 1)
    N, L = AllFeature.shape
    res = np.zeros([C, K])
    cnt = np.zeros([C, L])

    index = np.argsort(AllFeature, axis=1)[:,::-1][:,:k]
    for i in range(N):
        cnt[int(label[i])][index[i]] += 1
        
    res = np.argsort(cnt, axis=1)[:,::-1][:,start:start+K]
    return res


def sub_lda_select_feature(x_y):
    X, Y = x_y
    clean = X[Y==1]
    noisy = X[Y!=1]
    Si = np.mean((clean - clean.mean(axis=0, keepdims=True))*(clean - clean.mean(axis=0, keepdims=True)), axis=0) + np.mean((noisy - noisy.mean(axis=0, keepdims=True))*(noisy - noisy.mean(axis=0, keepdims=True)), axis=0)
    Sb = (clean.mean(axis=0) - noisy.mean(axis=0)) * (clean.mean(axis=0) - noisy.mean(axis=0)) 
    Sw = Sb/Si
    return np.argsort(Sw)[::-1]
    #return Sw

def lda_select_feature(feature, gt_label):
    X = []
    Y = []
    cate_num = int(np.max(gt_label))+1
    for i in range(cate_num):
        class_index = gt_label[:,1] == i
        x = feature[class_index]
        y = (gt_label[class_index,0] == gt_label[class_index,1]).astype(int)
        X.append(x)
        Y.append(y)
    svm_set = np.zeros([cate_num,2048])
    x_y = zip(X,Y)
    with mp.Pool(16) as pool:
        for i, c  in enumerate(pool.imap(sub_lda_select_feature, x_y, chunksize=8)):
           svm_set[i] = c
    return svm_set.astype(int)




def select_feature_orcle(all_feature,gtLabel,k):
    svm_set = lda_select_feature(all_feature, gtLabel)
    svm_set = svm_set[:,:k]
    return svm_set


def count_feature(regular_feature, output, groundtruth, k = 30, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 groundtruth [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    res = np.zeros([N])
    threshold_each_class = np.zeros([C])
    groundtruth = groundtruth.astype(int)

    index = np.argsort(output, axis=-1)[:,::-1][:,:k]

    for i in range(N):
        _, label = groundtruth[i]
        label = int(label)
        res[i] = np.intersect1d(index[i], regular_feature[label]).shape[0]

    for i in range(C):
        index_a_class = groundtruth[:,1] == i
        res_a_class = res[index_a_class]
        threshold_each_class[i] = np.quantile(res_a_class,tau,interpolation='lower')

    return res, threshold_each_class
def count_feature_single_GMM(regular_feature, output, groundtruth, k = 30, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 groundtruth [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    groundtruth = groundtruth.astype(int)
    res = np.zeros([N]).astype(bool)
    index = np.argsort(output, axis=-1)[:,::-1][:,:k]

    for i in range(N):
        _, label = groundtruth[i]
        label = int(label)
        cnt[i] = np.intersect1d(index[i], regular_feature[label]).shape[0]

    for i in range(C):
        index_a_class = groundtruth[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)

    return res

def L1_selection_noclean(regular_feature, output, gtLabel, isclean, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 gtLabel [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    output = output/np.linalg.norm(output, axis=1, keepdims=True)
    short_vec = np.zeros([C,K])
    for i in range(C):
        sub_index = (gtLabel[:,1] == i) * isclean
        short_vec[i] = output[sub_index].mean(axis=0)[regular_feature[i]]


    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    gtLabel = gtLabel.astype(int)
    res = np.zeros([N]).astype(bool)
 
    for i in range(N):
        _, label = gtLabel[i]
        label = int(label)
        if label <0: # missing sample
            cnt[i] = 0
        else:
            cnt[i] = np.linalg.norm(output[i][regular_feature[label]]- short_vec[label],1)

    for i in range(C):
        index_a_class = gtLabel[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob<tau)

    return res


def L2_selection_noclean(regular_feature, output, gtLabel, isclean, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 gtLabel [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    output = output/np.linalg.norm(output, axis=1, keepdims=True)
    short_vec = np.zeros([C,K])
    for i in range(C):
        sub_index = (gtLabel[:,1] == i) * isclean
        short_vec[i] = output[sub_index].mean(axis=0)[regular_feature[i]]


    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    gtLabel = gtLabel.astype(int)
    res = np.zeros([N]).astype(bool)
 
    for i in range(N):
        _, label = gtLabel[i]
        label = int(label)
        if label <0: # missing sample
            cnt[i] = 0
        else:
            cnt[i] = np.linalg.norm(output[i][regular_feature[label]]- short_vec[label],2)

    for i in range(C):
        index_a_class = gtLabel[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob<tau)

    return res

def cos_selection_noclean(regular_feature, output, gtLabel, isclean, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 gtLabel [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    output = output/np.linalg.norm(output, axis=1, keepdims=True)
    short_vec = np.zeros([C,K])
    for i in range(C):
        sub_index = (gtLabel[:,1] == i) * isclean
        short_vec[i] = output[sub_index].mean(axis=0)[regular_feature[i]]


    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    gtLabel = gtLabel.astype(int)
    res = np.zeros([N]).astype(bool)
 
    for i in range(N):
        _, label = gtLabel[i]
        label = int(label)
        if label <0: # missing sample
            cnt[i] = 0
        else:
            cnt[i] = np.sum(output[i][regular_feature[label]] * short_vec[label])

    for i in range(C):
        index_a_class = gtLabel[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)

    return res

def pca_selection_noclean(feature, gtLabel, isclean, k=200, tau = 0.5):
    N = feature.shape[0]
    C = int(gtLabel.max()+1)
    assert C == 200, C
    new_feature = np.ones([N,k])
    for i in range(C):
        cur_class = (gtLabel[:,1]==i)
        pca = PCA(n_components=k)
        new_feature[cur_class] = pca.fit_transform(feature[cur_class])
    new_feature = new_feature/(np.linalg.norm(new_feature, axis=1, keepdims=True)+1e-8)

    short_vec = np.zeros([C,k])
    for i in range(C):
        sub_index = (gtLabel[:,1] == i) * isclean
        if sub_index.sum()==0:
            short_vec[i] = 0
        else:
            short_vec[i] = new_feature[sub_index].mean(axis=0)
    
    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    gtLabel = gtLabel.astype(int)
    res = np.zeros([N]).astype(bool)
 
    for i in range(N):
        _, label = gtLabel[i]
        label = int(label)
        if label <0: # missing sample
            cnt[i] = 0
        else:
            cnt[i] = np.sum(new_feature[i] * short_vec[label])

    for i in range(C):
        index_a_class = gtLabel[:,1] == i
        res_a_class = cnt[index_a_class]
        
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= (threshold_each_class+1e-5)
    cnt[cnt > 1e5] = 1e5
    cnt[cnt < -1e5] = -1e5
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)

    return res

def cos_selection(regular_feature, output, groundtruth, k = 30, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 groundtruth [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    clean_set_size = 5 # from [1,40]
    clean_set_index = np.linspace(N-C*40, N-40, C).reshape((-1,1))
    clean_set_index = np.tile(clean_set_index, [1, clean_set_size]) + np.linspace(0, clean_set_size-1, clean_set_size).reshape((1, -1))
    clean_set_index = clean_set_index.reshape((-1)).astype(int)

    output = output/np.linalg.norm(output, axis=1, keepdims=True)
    short_vec = np.zeros([C,K])
    for i in range(C):
        sub_index = clean_set_index[i*clean_set_size:(i+1)*clean_set_size]
        short_vec[i] = output[sub_index].mean(axis=0)[regular_feature[i]]


    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    groundtruth = groundtruth.astype(int)
    res = np.zeros([N]).astype(bool)
 
    for i in range(N):
        _, label = groundtruth[i]
        label = int(label)
        if label <0: # missing sample
            cnt[i] = 0
        else:
            cnt[i] = np.sum(output[i][regular_feature[label]] * short_vec[label])

    for i in range(C):
        index_a_class = groundtruth[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)

    return res

def count_feature_single_GMM_weight(regular_feature, output, groundtruth, k = 30, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 groundtruth [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    groundtruth = groundtruth.astype(int)
    res = np.zeros([N]).astype(bool)
    index = np.argsort(output, axis=-1)[:,::-1][:,:k]
    feature_weight = []
    for i in range(L):
        feature_weight.append(np.log(C/np.sum(regular_feature==i)) if np.sum(regular_feature==i) > 0 else 0)

    for i in range(N):
        _, label = groundtruth[i]
        label = int(label)
        for term in index[i]:
            cnt[i] += feature_weight[term] if term in regular_feature[label] else 0

    for i in range(C):
        index_a_class = groundtruth[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)

    return res


def sub_feature_main_sub(regular_feature, index):
    length = np.zeros(200)
    for i in range(regular_feature.shape[0]):
        length[i] = np.intersect1d(index, regular_feature[i]).shape[0]
    return length

def count_feature_main_sub(regular_feature, output, groundtruth, k = 30, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 groundtruth [gt, label]
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape
    N, L = output.shape

    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    groundtruth = groundtruth.astype(int)
    res = np.zeros([N]).astype(bool)
    index = np.argsort(output, axis=-1)[:,::-1][:,:k]

    for i in range(N):
        _, label = groundtruth[i]
        label = int(label)
        cnt[i] = np.intersect1d(index[i], regular_feature[label]).shape[0]

    for i in range(C):
        index_a_class = groundtruth[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)

    print('main class selection')
    print('believe_clean {}'.format(np.sum(res)))
    print('actually clean {}'.format(np.sum(res*(groundtruth[:,0]==groundtruth[:,1]))))

    uncertain_sample = (prob>0.3)*(prob<0.7)
    sub_cnt = np.zeros(int(uncertain_sample.sum()))
    length = np.zeros([int(uncertain_sample.sum()), 200])
    func = partial(sub_feature_main_sub, regular_feature)


    with mp.Pool(16) as pool:
        for i, ll in enumerate(pool.imap(func, index[uncertain_sample], chunksize=16)):
            length[i] = ll
    for i in range(sub_cnt.shape[0]):
        label = groundtruth[i,1]
        length[i,label] = -1
        sub_cnt[i] = length[i].max()
    res[uncertain_sample] = sub_cnt<(sub_cnt.mean())

    return res

def sub_relabel_pq_dataset(regular_feature, index):
    new_label = np.zeros(regular_feature.shape[0])
    for i in range(regular_feature.shape[0]):
        current_length = np.intersect1d(index, regular_feature[i]).shape[0]
        new_label[i] = current_length
    new_label /= new_label.sum()
    return new_label

def optimize_L_sk(PS):

    tt = time.time()
    PS = PS.T # now it is K x N
    K, N = PS.shape
    r = np.ones((K, 1), dtype=np.float64) / K
    c = np.ones((N, 1), dtype=np.float64) / N
    PS **= 25  # K x N
    inv_K = np.float64(1./K)
    inv_N = np.float64(1./N)
    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
    # inplace calculations.
    PS *= np.squeeze(c)
    PS = PS.T
    PS *= np.squeeze(r)
    PS = PS.T
    argmaxes = np.nanargmax(PS, 0) # size N
    print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)
    return argmaxes

def relabel_pq_dataset(regular_feature, output, gtLabel, k=100, tau=0.5):
    C, K = regular_feature.shape
    N, L = output.shape

    new_label = np.zeros(N)-1

    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    gtLabel = gtLabel.astype(int)
    res = np.zeros([N]).astype(bool)
    index = np.argsort(output, axis=-1)[:,::-1][:,:k]

    for i in range(N):
        _, label = gtLabel[i]
        label = int(label)
        cnt[i] = np.intersect1d(index[i], regular_feature[label]).shape[0]
    threshold_each_class = np.zeros([C])
    for i in range(C):
        index_a_class = gtLabel[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class[i] = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class[i]
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)
    print("video cnt = {} in 1st round selection".format(np.sum(res)))

    new_label[res] = gtLabel[res,1]
    print('correct {}'.format(np.sum(new_label == gtLabel[:,0])))

    fake_label = np.zeros([np.sum(~res),C])
    func = partial(sub_relabel_pq_dataset, regular_feature)
    with mp.Pool(16) as pool:
        for i, fl in enumerate(pool.imap(func, index[~res], chunksize=16)):

            fake_label[i] = fl
    #assert (new_label!=-1).all()
    
    new_label[~res] =  optimize_L_sk(fake_label)
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    

    print("video cnt = {} in 2nd round selection".format(np.sum(new_label!=-1)))
    return new_label


def sub_relabel_dataset(regular_feature, index):
    new_label = 0
    max_length = 0
    for i in range(regular_feature.shape[0]):
        current_length = np.intersect1d(index, regular_feature[i]).shape[0]
        if current_length > max_length:
            max_length = current_length
            new_label = i
    return int(new_label), max_length

def relabel_dataset(regular_feature, output, gtLabel, k=100, tau=0.5):
    C, K = regular_feature.shape
    N, L = output.shape

    new_label = np.zeros(N)-1

    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    gtLabel = gtLabel.astype(int)
    res = np.zeros([N]).astype(bool)
    index = np.argsort(output, axis=-1)[:,::-1][:,:k]

    for i in range(N):
        _, label = gtLabel[i]
        label = int(label)
        cnt[i] = np.intersect1d(index[i], regular_feature[label]).shape[0]
    threshold_each_class = np.zeros([C])
    for i in range(C):
        index_a_class = gtLabel[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class[i] = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class[i]
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)
    print("video cnt = {} in 1st round selection".format(np.sum(res)))
    cnt = cnt.reshape([-1])
    new_label[res] = gtLabel[res,1]
    func = partial(sub_relabel_dataset, regular_feature)
    with mp.Pool(16) as pool:
        for i, (nl,ml) in enumerate(pool.imap(func, index, chunksize=16)):
            if not res[i]:
                new_label[i], cnt[i] = nl,ml/threshold_each_class[nl]
            if ml == 0:
                print('ml = 0')
    #assert (new_label!=-1).all()
    
    tau_second = np.percentile(cnt[~res],80)
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    new_label[(cnt<tau_second) * (~res)] = -1

    print("video cnt = {} in 2nd round selection".format(np.sum(new_label!=-1)))
    return new_label

def relabel_pre_dataset(regular_feature, output, gtLabel, k=100, tau=0.5):
    C, K = regular_feature.shape
    N, L = output.shape

    new_label = np.zeros(N)-1

    cnt = np.zeros([N])
#    threshold_each_class = np.zeros([C])
    gtLabel = gtLabel.astype(int)
    res = np.zeros([N]).astype(bool)
    index = np.argsort(output, axis=-1)[:,::-1][:,:k]

    for i in range(N):
        _, label = gtLabel[i]
        label = int(label)
        cnt[i] = np.intersect1d(index[i], regular_feature[label]).shape[0]
    threshold_each_class = np.zeros([C])
    for i in range(C):
        index_a_class = gtLabel[:,1] == i
        res_a_class = cnt[index_a_class]
        threshold_each_class[i] = np.mean(res_a_class)
        cnt[index_a_class] /= threshold_each_class[i]
    cnt = cnt.reshape([-1, 1])    
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(cnt)
    prob = GMM.predict_proba(cnt)
    prob = prob[:,GMM.means_.argmax()]
#    print("AUC = {}".format(roc_auc_score(y_true, y_scores)))
    res = (prob>tau)
    print("video cnt = {} in 1st round selection".format(np.sum(res)))
    new_label[res] = gtLabel[res,1]
    print('correct {}'.format(np.sum(new_label==gtLabel[:,0])))
    


    return new_label









def feature_recorrection(regular_feature, output, k = 30, tau = 0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    Output: N res
    C threshold_each_class
    '''
    C, K = regular_feature.shape[0], regular_feature.shape[1]
    N = output.shape[0]
    index = np.argsort(output, axis=-1)[:,::-1][:,:k]
    res = np.zeros(N) - 1
  #  print(index[0])
  #  print(regular_feature[0])
    for i in range(N):
        ratio = np.zeros(C)
        for j in range(C):
            ratio[j] = np.intersect1d(index[i], regular_feature[j]).shape[0] / k
        if np.max(ratio) >= tau:
          #  print(np.max(ratio))
            res[i] = np.argmax(ratio)
          #  print(res[i])
    print(res)
    return res


def count_feature_GMM(regular_feature, output, groundtruth, k = 30, tau=0.5):
    '''
    Input: C * K regular_feature 
    N * L output
    N * 2 groundtruth [gt, label]
    Output: N res
    
    '''
    C, K = regular_feature.shape
    N, L = output.shape
    cnt = np.zeros([N])
    res = np.zeros([N]).astype(bool)
    threshold_each_class = np.zeros([C])
    groundtruth = groundtruth.astype(int)

    index = np.argsort(output, axis=-1)[:,::-1][:,:k]

    for i in range(N):
        _, label = groundtruth[i]
        label = int(label)
        cnt[i] = np.intersect1d(index[i], regular_feature[label]).shape[0]
    
    cnt = cnt.reshape(-1,1)

    for i in range(C):
        index_a_class = groundtruth[:,1] == i
        res_a_class = cnt[index_a_class,:].reshape(-1,1)
        GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        GMM.fit(res_a_class)
        prob = GMM.predict_proba(cnt)
        prob = prob[:,GMM.means_.argmax()]
        res += (prob>tau)*index_a_class
    #    threshold_each_class[i] = np.quantile(res_a_class,tau,interpolation='lower')

    return res

def get_init_params(all_loss, clean_set_index):
    
    label = KMeans(n_clusters=2, n_init=1).fit(all_loss).labels_
    init_mean = np.zeros([2])
    init_precision = np.zeros([2])

    noisy_label = 0 if np.mean(all_loss[label==0]) > np.mean(all_loss[label==1]) else 1
    init_mean[0] = np.mean(all_loss[clean_set_index])
    init_mean[1] = np.mean(all_loss[label==noisy_label])

    init_precision[0] = 1/(np.var(all_loss[clean_set_index])+1e-8)
    init_precision[1] = 1/(np.var(all_loss[label==noisy_label])+1e-8)

    init_mean = init_mean.reshape(2,1)
    init_precision = init_precision.reshape(2,1,1)

    return init_mean, init_precision
    # elif use a GMM for all classes
def get_split_by_loss_clean(init_mean, init_precision, all_loss, tau=0.5):
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4, means_init=init_mean, precisions_init = init_precision)
    GMM.fit(all_loss)
    prob = GMM.predict_proba(all_loss)
    prob = prob[:,GMM.means_.argmin()]
    res = (prob>tau)

    return res
def get_split_by_loss(all_loss, tau=0.5):
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(all_loss)
    prob = GMM.predict_proba(all_loss)
    prob = prob[:,GMM.means_.argmin()]
    res = (prob>tau)

    return res

def get_split_by_loss_prob(all_loss):
    GMM = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    GMM.fit(all_loss)
    prob = GMM.predict_proba(all_loss)
    prob = prob[:,GMM.means_.argmin()]
   # res = (prob>tau)

    return prob


def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, ind_1_update
if __name__ == "__main__":


    f = np.random.random([78560,2048])
    gtlabel = np.zeros([78560,2])
    gtlabel[:,1] = (np.linspace(1,78560,78560) % 200).astype(int)
    print(gtlabel.shape)
    isclean = np.ones([78560]).astype(bool)
    s = time.time()
    pca_selection_noclean(f, gtlabel, isclean)
    print(time.time()-s)








    
    





    
