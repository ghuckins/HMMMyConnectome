import numpy as np
import os
from scipy.stats import zscore
import pandas as pd
from hmm import get_saved_params, get_key, fit_all_models, init_transonly, logprob_all_models
from k_means import kmeans_init
import random
import jax.numpy as jnp
from jax import vmap
import pickle
import warnings
warnings.filterwarnings("ignore")

from src.dynamax.hidden_markov_model.models.gaussian_hmm import DiagonalGaussianHMM
from src.dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM

root = "/Users/gracehuckins/PycharmProjects/HMMMyConnectome"
data_root = "/Users/gracehuckins/Documents/Research Data"


def import_hcp(num_networks, heldout=False):
    """
    Loads the preprocessed HCP data into a dictionary

    Args:
        num_networks: int (7 or 17) indicating whether to import 7- or 17-network parcellation

    Returns:
        data: a dict where each key is a subj. identifier and each value is a num_timepoints x num_networks numpy array
    """
    if heldout:
        path = os.path.join(root, "data", f"hcpdata{num_networks}_heldout")
    else:
        path = os.path.join(root, "data", f"hcpdata{num_networks}")
    if not os.path.exists(path):
        import_raw_hcp(num_networks)
    data = {}
    for file in os.listdir(path):
        file_key = int(file[3:9])
        if file_key in data.keys():
            data[file_key].append(np.loadtxt(os.path.join(path,file)))
        else:
            data[file_key] = [np.loadtxt(os.path.join(path,file))]
    return data


def alt_params(data, latdim, trans=False, ar=False):
    '''
    Retrieves previously saved model parameters fit to 3/4 scans for each subject in the data dict,
    or fits new parameters and saves them if they don't already exist

    Args:
        data: a dict of lists of num_timepoints x num_networks numpy arrays, keyed by subject identifier
        latdim: number of hidden states for the HMM
        trans: whether to fit a transition matrix-only model
        ar: whether to fit an autoregressive model

    Returns:
        params: a dict of fit parameters for each subject
    '''

    obsdim = np.shape(list(data.values())[0])[2]

    ar_str = ""
    trans_str = ""
    if ar:
        ar_str = "ar"
    if trans:
        trans_str = "trans"
    dir = os.path.join(root, "results", f"hcpparams{ar_str}{trans_str}{obsdim}")

    if not os.path.exists(dir):
        os.mkdir(dir)

    filepath = os.path.join(dir, f"params{latdim}")
    if not os.path.exists(filepath):
        print(f"fitting params {latdim} " + ar_str + " " + trans_str)
        params = {}

        if trans:
            emissions, probs = get_saved_params(obsdim, latdim, ar=ar, key_string="hcp")
            hmm, base_params, props = init_transonly(emissions, probs, ar=ar)
        else:
            if ar:
                hmm = LinearAutoregressiveHMM(latdim, obsdim)
            else:
                hmm = DiagonalGaussianHMM(latdim, obsdim)

        for key in data.keys():
            params[key] = []
            for i in range(len(data[key])):
                temp_data = data[key].copy()
                temp_data.pop(i)
                if not trans:
                    base_params, props = kmeans_init(hmm, jnp.array(temp_data), get_key(), ar=ar)
                curr_params = fit_all_models(hmm, base_params, props, jnp.array(temp_data), ar=ar)
                params[key].append(curr_params)

        with open(filepath, "wb") as file:
            pickle.dump(params, file)

    else:
        with open(filepath,"rb") as file:
            params = pickle.load(file)

    return params


def loohcp(data, latdim, num_subjs, trans=False, ar=False, lags=1):
    """
    Performs leave-one-out cross-validation by fitting individual HMMs to 3 runs from each subject in data,
    and then evaluating whether the 4th run from each subject has the maximum log likelihood under that subject's HMM

    Args:
        data: a dict of num_timepoints x num_networks numpy arrays, keyed by subject identifier
        latdim: number of hidden states for the HMM
        num_subjs: how many subjects from the dict to use for the cross-validation
        trans: whether to fit a transition matrix-only model
        ar: whether to fit an autoregressive model
        lags: if the model is autoregressive, how many lags to include

    Returns:
        The accuracy of the classifier (chance is 1/num_subjs)
    """
    obsdim = np.shape(list(data.values())[0])[2]

    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    params = {}

    if trans:
        emissions, probs = get_saved_params(obsdim, latdim, ar=ar, key_string="hcp")
        hmm, base_params, props = init_transonly(emissions, probs, ar=ar)

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

    correct = 0
    for key in keys:
        train_data = data[key]
        random.shuffle(train_data)
        train_data = train_data[:-1]
        if not trans:
            base_params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(train_data))
        params[key] = fit_all_models(hmm, base_params, props, np.array(train_data), ar=ar)
    for key in keys:
        i = 0
        while i < len(data[key]):
            train_data = data[key].copy()
            test = train_data.pop(i)
            if not trans:
                base_params, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(train_data))
            loo_params = fit_all_models(hmm, base_params, props, np.array(train_data), ar=ar)
            log_likelihoods = [logprob_all_models(hmm, loo_params, test, ar=ar)]
            for key2 in keys:
                if key2 != key:
                    log_likelihoods.append(logprob_all_models(hmm, params[key2], test, ar=ar))
            if np.argmax(log_likelihoods) == 0:
                correct += 1
            i = i + 1

    return correct/(num_subjs*4)


def loohcp_1state(data, num_subjs, ar=True, lags=1):
    """
    DELETE AFTER TESTING??

    Performs leave-one-out cross-validation by fitting individual HMMs to 3 runs from each subject in data,
    and then evaluating whether the 4th run from each subject has the maximum log likelihood under that subject's HMM

    Args:
        data: a dict of num_timepoints x num_networks numpy arrays, keyed by subject identifier
        latdim: number of hidden states for the HMM
        num_subjs: how many subjects from the dict to use for the cross-validation
        trans: whether to fit a transition matrix-only model
        ar: whether to fit an autoregressive model
        lags: if the model is autoregressive, how many lags to include

    Returns:
        The accuracy of the classifier (chance is 1/num_subjs)
    """
    obsdim = np.shape(list(data.values())[0])[2]

    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    params = {}

    if ar:
        hmm = LinearAutoregressiveHMM(1, obsdim, num_lags=lags)
    else:
        hmm = DiagonalGaussianHMM(1, obsdim)

    correct = 0
    for key in keys:
        train_data = data[key]
        random.shuffle(train_data)
        train_data = train_data[:-1]
        base_params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(train_data), initial_probs=np.array([1.]), transition_matrix=np.array([[1.]]))
        props.transitions.transition_matrix.trainable = False
        props.initial.probs.trainable = False
        params[key] = fit_all_models(hmm, base_params, props, np.array(train_data), ar=ar)
    for key in keys:
        i = 0
        while i < len(data[key]):
            train_data = data[key].copy()
            test = train_data.pop(i)
            base_params, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(train_data), initial_probs=np.array([1.]), transition_matrix=np.array([[1.]]))
            loo_params = fit_all_models(hmm, base_params, props, np.array(train_data), ar=ar)
            log_likelihoods = [logprob_all_models(hmm, loo_params, test, ar=ar)]
            for key2 in keys:
                if key2 != key:
                    log_likelihoods.append(logprob_all_models(hmm, params[key2], test, ar=ar))
            if np.argmax(log_likelihoods) == 0:
                correct += 1
            i = i + 1

    return correct/(num_subjs*4)


def baseline_fingerprint(num_networks, num_subjs):
    """
    Uses correlation of correlations to classify runs of data according to subject identity

    Args:
        num_networks: int (7, 17, or 512) indicating whether to use 7- or 17-network Yeo parcellation;
        or, if 512, unparcellated data
        num_subjs: how many subjects to use for the cross-validation

    Returns:
        The accuracy of the classifier
    """
    data = import_hcp(num_networks)
    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    compare_cov = {}
    for key in keys: #getting baseline average cov matrices from 3 runs for all participants
        compare_data = data[key]
        random.shuffle(compare_data)
        compare_data = compare_data[:-1]
        compare_cov[key] = np.average(np.array([np.corrcoef(item.T) for item in compare_data]),axis=0)
    correct = 0
    for key in keys:
        i = 0
        while i < len(data[key]):
            train_data = data[key].copy()
            train_data.pop(i)
            train_cov = np.average(np.array([np.corrcoef(item.T) for item in train_data]),axis=0)
            cov = np.corrcoef(data[key][i].T).flatten()
            all_corr = [np.corrcoef(train_cov.flatten(), cov)[0,1]]
            j = 0
            while j < len(keys):
                corr = np.corrcoef(compare_cov[keys[j]].flatten(), cov)[0,1]
                if keys[j] != key:
                    all_corr.append(corr)
                j = j + 1
            if np.argmax(all_corr) == 0:
                correct += 1
            i = i + 1

    return correct/(4*num_subjs)


def main():
    import_raw_hcp(512)


if __name__ == "__main__":
    main()
