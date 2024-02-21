import numpy as np
import os
from scipy.stats import zscore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from hmm import gsr, get_network_activity, get_saved_params, get_key,\
    fit_all_models, init_transonly, logprob_all_models, get_params
from k_means import kmeans_init
import random
import jax.numpy as jnp
from jax import vmap
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

from src.dynamax.hidden_markov_model.models.gaussian_hmm import DiagonalGaussianHMM
from src.dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM

root = "/Users/gracehuckins/PycharmProjects/HMMMyConnectome"
data_root = "/Users/gracehuckins/Documents/Research Data"

def import_raw_hcp(num_networks):
    """
    Imports and preprocesses HCP data by:
        -applying global signal regression
        -averaging activity across Yeo networks (according to parcellation specified by num_networks)
        -z-scoring activity across time for each network

    Args:
        num_networks: int (7 or 17) indicating whether to use 7- or 17-network Yeo parcellation

    Returns:
        None
    """
    path = os.path.join(data_root, "HCP")
    files = {}
    for file in os.listdir(path): #sorting data by subject identifier
        #the point of having this here now is to check that each subj has 4 runs
        #is there a more efficient way? need to check how # of runs is coded
        #like if i can check for existence of a 4th run
        file_key = file[4:10]
        if file_key in files.keys():
            files[file_key].append(file)
        else:
            files[file_key] = [file]

    #I want to fix all of this held-out stuff to hold out raw data according to same pattern by which data was previously  held out
    #FIX FIX FIX FIX

    savepath = os.path.join(root, "data", f"hcpdata{num_networks}")

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for key in files.keys():
        if len(files[key]) < 4: #if I don't have 4 runs for that subject
            continue
        i = 0
        while i < 4:
            raw_data = np.load(os.path.join(path, files[key][i]))
            raw_data = gsr(raw_data)
            if num_networks == 512:
                activities = zscore(raw_data, axis=0)
            else:
                activities = zscore(get_network_activity(raw_data, num_networks, hcp=True), axis=0)
            np.savetxt(os.path.join(savepath, f"sub{key}_{i}.txt"), activities)
            i += 1

    return None


def import_hcp(num_networks):
    """
    Loads the preprocessed HCP data into a dictionary

    Args:
        num_networks: int (7 or 17) indicating whether to import 7- or 17-network parcellation

    Returns:
        data: a dict where each key is a subj. identifier and each value is a num_timepoints x num_networks numpy array
    """
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


def import_randomized(num_networks):
    """

    TO DELETE AFTER TESTING

    """
    path = os.path.join(root, "data", f"hcpdata{num_networks}")
    if not os.path.exists(path):
        import_raw_hcp(num_networks)
    data = {}
    keys = list(import_hcp(num_networks).keys())
    for key in keys:
        data[key] = []
    for file in os.listdir(path):
        file_key = random.choice(keys)
        while len(data[file_key]) == 4:
            file_key = random.choice(keys)
        data[file_key].append(np.loadtxt(os.path.join(path,file)))
    return data


def alt_params(data, latdim, trans=False, ar=False):
    '''
    STILL TO FIX

    Args:
        data:
        latdim:
        trans:
        ar:

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

    filepath = os.path.join(dir,f"params{latdim}")
    if not os.path.exists(filepath):
        print(f"fitting params {latdim}{trans}{ar}")
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


def loohcp_confusion(data, latdim, num_subjs, trans=False, ar=False, lags=1):
    """
    TO DELETE OR REPLACE

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

    confusion = pd.DataFrame(data=np.zeros((num_subjs,num_subjs)), index=keys, columns=keys)

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
            log_likelihood = logprob_all_models(hmm, loo_params, test, ar=ar)
            best_key = key
            for key2 in keys:
                if key2 != key:
                    if log_likelihood <= logprob_all_models(hmm, params[key2], test, ar=ar):
                        best_key = key2
            confusion[key][best_key] += 1
            if key == best_key:
                correct += 1
            i = i + 1

    return correct/(num_subjs*4), confusion


def loohcp_batch(data, latdim, num_subjs, trans=False, ar=False, lags=1):
    """
    Performs leave-one-out cross-validation by fitting individual HMMs to 3 runs from each subject in data,
    and then evaluating whether the 4th run from each subject has the maximum log likelihood under that subject's HMM
    in a batched manner that takes advantage of Jax parallelization

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

    keys = list(data.keys()) #list of keys for all subjects in dataset
    random.shuffle(keys)
    keys = keys[:num_subjs] #randomly select num_subjs subjects
    params = alt_params(data, latdim, trans, ar)

    if trans:
        emissions, probs = get_saved_params(obsdim, latdim, ar=ar, key_string="hcp")
        hmm, b_p, pr = init_transonly(emissions, probs, ar=ar)

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

    train = []
    test = []
    for key in keys:
        curr_data = data[key]
        test.extend(curr_data)
        curr_data = jnp.array(curr_data)
        train.extend([jnp.concatenate([curr_data[:i], curr_data[i + 1:]]) for i in range(4)])

    train = jnp.stack(train)
    test = jnp.stack(test)

    def _fit_fold(train, test, alt_params):
        #here we test all testing data against all alternative models, including model trained on that subjects' data.
        #that just made it easier to code up—we account for tihs in calculating accuracy below.
        if trans:
            base_params = b_p
            props = pr
        else:
            base_params, props = kmeans_init(hmm, train, get_key(), ar=ar)
        loo_params = fit_all_models(hmm, base_params, props, train, ar=ar)
        ll = [logprob_all_models(hmm, loo_params, test, ar=ar)]
        ll.extend([logprob_all_models(hmm, alt_params[i], test, ar=ar) for i in range(num_subjs)])
        return jnp.argsort(-jnp.array(ll))

    param_list = [random.choice(params[keys[i]]) for i in range(num_subjs)]
    ll_sort = vmap(_fit_fold, in_axes=[0, 0, None])(train, test, param_list)
    indices = jnp.repeat(jnp.arange(num_subjs), 4)+1
    #2 ways to get correct classification: either the model fit by the other 3 runs on that subject did best (first
    #element in argsort should be 0) or it did second best, after the alt_params model fit on that subjects' data, which
    #we didn't really want to fit on anyway. next line of code takes into account those 2 possibilities.
    correct = jnp.sum((ll_sort[:, 0] == 0).astype(int)) + jnp.dot(jnp.equal(indices,ll_sort[:,0]).astype(int), (ll_sort[:,1] == 0).astype(int))
    return correct/(num_subjs*4)


def main():
    data = import_hcp(7)
    reps = 10
    batch = 0
    baseline = 0
    for rep in range(reps):
        batch += loohcp_batch(data, 6, 10, ar=False, trans=False)
        baseline += loohcp(data, 6, 10, ar=False, trans=False)
    print(f"batch acc: {batch/reps}")
    print(f"baseline acc: {baseline/reps}")

    quit()

    alt_params(data, 6, trans=False, ar=False)
    quit()




def baseline_fingerprint(num_networks, num_subjs):
    """
    Uses correlation of correlations to classify runs of data according to subject identity;
    accuracy evaluated via leave-one-out cross-validation

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


if __name__ == "__main__":
    main()
