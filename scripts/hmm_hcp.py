import numpy as np
import os
import pandas as pd
import random
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn import svm
import math
import matplotlib.pyplot as plt
from hmm import gsr, getNetworkActivity, fit_model_transonly

from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

root = "/Users/gracehuckins/Documents/HMMMyConnectome"

def import_raw_hcp(num_networks):
    path = os.path.join(root,"data/HCP")
    files = {}
    for file in os.listdir(path):
        file_key = file[4:10]
        if file_key in files.keys():
            files[file_key].append(file)
        else:
            files[file_key] = [file]
    counter = 0
    data_dir = os.path.join(root,"results",f"hcpdata{num_networks}")
    held_out_dir = os.path.join(root, "results", f"hcpdata{num_networks}heldout")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(held_out_dir):
        os.mkdir(held_out_dir)
    for key in files.keys():
        if len(files[key]) < 4:
            continue
        if counter % 4 == 0:
            save_dir = held_out_dir
        else:
            save_dir = data_dir
        i = 0
        while i < 4:
            raw_data = np.load(os.path.join(path, files[key][i]))
            raw_data = gsr(raw_data)
            activities = zscore(getNetworkActivity(raw_data, num_networks, True), axis=0)
            np.savetxt(os.path.join(save_dir, f"sub{key}_{i}.txt"), activities)
            i += 1
        counter = counter + 1

def load_hcp(num_networks):
    dir = os.path.join(root,"results",f"hcpdata{num_networks}")
    if not os.path.exists(dir):
        import_raw_hcp(num_networks)
    data = {}
    for file in os.listdir(dir):
        file_key = file[3:9]
        if file_key in data.keys():
            data[file_key].append(np.loadtxt(os.path.join(dir,file)))
        else:
            data[file_key] = [np.loadtxt(os.path.join(dir,file))]
    return data

def loohcp(num_networks, latdim, num_subjs):
    '''
    Performs LOO cross-validation on HCP data

    Args:
        num_subjs: number of subjects to use

    Returns: Balanced accuracy of the classifier

    '''
    data = load_hcp(num_networks)
    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    hmm = DiagonalGaussianHMM(latdim, num_networks)
    params = {}
    correct = 0
    for key in keys:
        train_data = data[key]
        random.shuffle(train_data)
        train_data = train_data[:-1]
        temp_params, props = hmm.initialize(method="kmeans", emissions=np.array(train_data))
        params[key], _ = hmm.fit_em(temp_params, props, np.array(train_data), num_iters=100, verbose=False)
    for key in keys:
        i = 0
        while i < len(data[key]):
            train_data = data[key].copy()
            train_data.pop(i)
            temp_params, _ = hmm.initialize(method="kmeans", emissions=np.array(train_data))
            loo_params, _ = hmm.fit_em(temp_params, props, np.array(train_data), num_iters=100, verbose=False)
            log_likelihoods = [hmm.marginal_log_prob(loo_params, data[key][i])]
            for key2 in keys:
                if key2 != key:
                    log_likelihoods.append(hmm.marginal_log_prob(params[key2], data[key][i]))
            if np.argmax(log_likelihoods) == 0:
                correct += 1
            i = i + 1
    return correct/(num_subjs*4)

def loohcp_ar(num_networks, latdim, lags, num_subjs):
    '''
    Performs LOO cross-validation on HCP data with an HMM-MAR model.

    Args:
        num_subjs: number of subjects to use

    Returns: Balanced accuracy of the classifier

    '''
    data = load_hcp(num_networks)
    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    hmm = LinearAutoregressiveHMM(latdim, num_networks, num_lags=lags)
    params = {}
    correct = 0
    for key in keys:
        train_data = data[key]
        random.shuffle(train_data)
        train_data = train_data[:-1]
        temp_params, props = hmm.initialize(method="kmeans", emissions=np.array(train_data))
        inputs = []
        for datum in train_data:
            inputs.append(hmm.compute_inputs(datum))
        params[key], _ = hmm.fit_em(temp_params, props, np.array(train_data), inputs=np.array(inputs), num_iters=100, verbose=False)
    for key in keys:
        i = 0
        while i < len(data[key]):
            train_data = data[key].copy()
            train_data.pop(i)
            random.shuffle(train_data)
            temp_params, _ = hmm.initialize(method="kmeans", emissions=np.array(train_data))
            inputs = []
            for datum in train_data:
                inputs.append(hmm.compute_inputs(datum))
            loo_params, _ = hmm.fit_em(temp_params, props, np.array(train_data), inputs=np.array(inputs), num_iters=100, verbose=False)
            inputs = hmm.compute_inputs(data[key][i])
            log_likelihoods = [hmm.marginal_log_prob(loo_params, data[key][i], inputs=inputs)]
            for key2 in keys:
                if key2 != key:
                    log_likelihoods.append(hmm.marginal_log_prob(params[key2], data[key][i], inputs=inputs))
            if np.argmax(log_likelihoods) == 0:
                correct += 1
            i = i + 1
    return correct/(num_subjs*4)

def loohcptrans(num_networks, num_subjs, latdim):
    '''
    Performs LOO cross-validation on HCP data

    Args:
        num_subjs: number of subjects to use

    Returns: Balanced accuracy of the classifier

    '''
    data = load_hcp(num_networks)
    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    params = {}
    correct = 0
    for key in keys:
        train_data = data[key]
        random.shuffle(train_data)
        train_data = train_data[:-1]
        _, params[key], _ = fit_model_transonly(train_data, latdim, hcp=True)
    for key in keys:
        i = 0
        while i < len(data[key]):
            train_data = data[key].copy()
            train_data.pop(i)
            hmm, loo_params, _ = fit_model_transonly(train_data, latdim, hcp=True)
            log_likelihoods = [hmm.marginal_log_prob(loo_params, data[key][i])]
            for key2 in keys:
                if key2 != key:
                    log_likelihoods.append(hmm.marginal_log_prob(params[key2], data[key][i]))
            if np.argmax(log_likelihoods) == 0:
                correct += 1
            i = i + 1
    return correct/(num_subjs*4)

print(loohcp(7,5,100))
print(loohcp_ar(7,5,1,100))