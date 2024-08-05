import numpy as np
import os
import pandas as pd
import random
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn import svm
import math
import matplotlib.pyplot as plt
import pickle
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from k_means import kmeans_init
import seaborn as sns
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore")

from src.dynamax.hidden_markov_model.models.gaussian_hmm import DiagonalGaussianHMM
from src.dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM

root = "/Users/gracehuckins/PycharmProjects/HMMMyConnectome"
data_root = "/Users/gracehuckins/Documents/Research Data"


def import_all(num_networks):
    """
    Imports all the MyConnectome data as a single list of numpy arrays

    Args:
        num_networks: int (7 or 17) indicating whether to import 7- or 17-network parcellation

    Returns:
        data: list of numpy arrays, each array is one recording
    """
    path = os.path.join(root, "data", f"data{num_networks}")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    data = []
    for file_name in os.listdir(path):
        data.append(np.loadtxt(os.path.join(path, file_name)))
    return data


def import_tuesthurs(num_networks, split=False):
    """
    Separately imports MyConnectome Tuesday and Thursday recordings into individual lists

    Args:
        num_networks: int (7 or 17) indicating whether to import 7- or 17-network parcellation
        split: whether to import split data (which consists only of continuous recordings below head-motion threshold)

    Returns:
        tues_data: a list of numpy arrays, each of which is a recording made on Tuesday
        thurs_data: a list of numpy arrays, each of which is a recording made on Thursday
    """
    if split:
        path = os.path.join(root, "data", f"data{num_networks}_split")
    else:
        path = os.path.join(root, "data", f"data{num_networks}")
    tues_data = []
    thurs_data = []
    for file_name in os.listdir(path):
        if file_name[6] == "t":
            tues_data.append(np.loadtxt(os.path.join(path, file_name)))
        if file_name[6] == "r":
            thurs_data.append(np.loadtxt(os.path.join(path, file_name)))
    return tues_data, thurs_data


def import_raw(num_networks):
    """
    Imports and preprocesses MyConnectome data by:
        -applying global signal regression
        -averaging activity across Yeo networks (according to parcellation specified by num_networks)
        -z-scoring activity across time for each network
        -appending a "t" or "r" to each filename to indicate whether the recording was made on Tuesday or Thursday

    Args:
        num_networks: int (7 or 17) indicating whether to use 7- or 17-network Yeo parcellation

    Returns:
        None
    """
    path = os.path.join(data_root, "MyConnectome")
    metadata = pd.read_table(os.path.join(path, "trackingdata_goodscans.txt"))
    metadata = metadata.set_index("subcode")
    savepath = os.path.join(root, "data", f"data{num_networks}")

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for filename in os.listdir(path):
        if filename.startswith("sub"):
            day = metadata.loc[filename.replace(".txt", "")]["day_of_week"]
            raw_data = np.loadtxt(os.path.join(path, filename))
            raw_data = gsr(raw_data)
            activities = zscore(get_network_activity(raw_data, num_networks), axis=0)

            if day == "2":
                np.savetxt(
                    os.path.join(savepath, filename.replace(".txt", "t.txt")),
                    activities,
                )
            elif day == "4":
                np.savetxt(
                    os.path.join(savepath, filename.replace(".txt", "r.txt")),
                    activities,
                )
            else:
                np.savetxt(os.path.join(savepath, filename), activities)

    return None


def get_network_activity(data, num_networks, hcp=False):
    """
    Averages activation across either the Yeo 7 or Yeo 17 networks at each timepoint in the data

    Args:
        data: a num_timepoints x num_ROIs numpy array of data
        num_networks: int (7 or 17) indicating whether to use 7- or 17-network Yeo parcellation
        hcp: whether the data come from the HCP (True) or MyConnectome (False) datasets

    Returns:
        A num_timepoints x num_networks numpy array of the network-averaged data
    """
    if hcp:
        parcellation = f"Yeo_networks{num_networks}"
        path = os.path.join(data_root, "HCP/parcel_data_hcp.txt")
    else:
        parcellation = f"{num_networks}networks"
        path = os.path.join(data_root, "MyConnectome/parcel_data.txt")
    parcels = pd.read_table(path)[parcellation]
    roughnetworks = pd.unique(parcels)
    if hcp:
        roughnetworks = np.delete(
            roughnetworks, np.where(roughnetworks == "No network found")
        )
    roughnetworks.sort() # to ensure a consistent order for the networks in averaged data
    print(roughnetworks)
    activities = []
    for network in roughnetworks:
        if network.lower().startswith(parcellation) or hcp:
            netactivity = np.average(data[:, (parcels == network)], axis=1).reshape(
                (-1, 1)
            )
            activities.append(netactivity)
    return np.concatenate(activities, axis=1)


def gsr(data):
    """
    Applies global signal regression to data

    Args:
        data: A numpy array of fMRI data; each row is a timepoint and each column is a ROI

    Returns:
        gsr_data: A numpy array of fMRI data with global signal regressed out
    """
    gsignal = np.average(data, axis=1)
    gsignal = np.reshape(gsignal, (-1, 1))
    ginverse = np.linalg.inv(gsignal.T @ gsignal) @ gsignal.T
    beta = ginverse @ data
    gsr_data = data - gsignal @ beta
    return gsr_data


def get_key():
    """
    Returns a random PRNG key in order to randomize model fitting and other stochastic Jax-based operations

    Returns: A PRNG key usable by Jax functions
    """
    return jr.PRNGKey(random.randint(0, 10000))


def get_params(data, latdim, ar=False, key_string="", lags=1):
    """
    Fits a chosen HMM to a dataset and saves the resulting HMM parameters, except the transition matrix, for later use

    Args:
        data: a list of num_timepoints x num_networks numpy arrays
        latdim: the number of hidden states in the model
        ar: whether or not the model is autoregressive
        key_string: "" if data are from MyConnectome, "hcp" if data are from HCP
        lags: if the model is autoregressive, how many lags to include

    Returns:
        None
    """
    obsdim = np.shape(data[0])[1]

    if ar:
        key_string = key_string + "ar"

    path = os.path.join(root, "results", f"{key_string}hmm{obsdim}")
    if os.path.exists(os.path.join(path, f"emissions{latdim}")): # checking if parameters have already been saved
        return None

    if ar:
        hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        inputs = jnp.stack([hmm.compute_inputs(datum) for datum in data])

    else:
        hmm = DiagonalGaussianHMM(latdim, obsdim)

    if not os.path.exists(path):
        os.mkdir(path)

    params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data))

    if ar:
        params, ll = hmm.fit_em(params, props, np.array(data), inputs=np.array(inputs), num_iters=500)
    else:
        params, ll = hmm.fit_em(params, props, np.array(data), num_iters=500)

    np.save(os.path.join(path, f"probs{latdim}"), params.initial.probs)

    with open(os.path.join(path, f"emissions{latdim}"), "wb") as file:
        pickle.dump(params.emissions, file)

    plt.plot(range(len(ll)), ll)
    plt.show()

    return None


def get_saved_params(obsdim, latdim, ar=False, key_string=""):
    """
    Retrieves previously saved fit parameters for a given HMM and dataset

    Args:
        obsdim: the emission dimension of the model
        latdim: the number of hidden states in the model
        ar: whether or not the model is autoregressive
        key_string: "" if previously fit data are from MyConnectome, "hcp" if data are from HCP

    Returns:
        emissions: the fit emissions parameters
        probs: the fit initial state probabilites
    """
    if ar:
        path = os.path.join(root, "results", f"{key_string}arhmm{obsdim}")
    else:
        path = os.path.join(root, "results", f"{key_string}hmm{obsdim}")

    assert os.path.exists(os.path.join(path, f"emissions{latdim}")), \
        "You need to save the parameters first using the get_params method."

    with open(os.path.join(path, f"emissions{latdim}"), "rb") as file:
        emissions = pickle.load(file)
    probs = np.load(os.path.join(path, f"probs{latdim}.npy"))

    return emissions, probs


def fit_all_models(hmm, params, props, data, ar=False, num_iters=100):
    """
    Given an HMM, its parameters and properties, and a dataset, fits the model to that dataset

    Args:
        hmm: a dynamax HMM object
        params: the parameters of the HMM
        props: the properties of the HMM's parameters, which dictate which parameters should be trained
        data: a list of num_timepoints x num_networks numpy arrays
        ar: whether or not the model is autoregressive
        num_iters: the number of EM iterations to carry out

    Returns:
        params: the fit parameters of the HMM
    """
    if ar:
        inputs = jnp.stack([hmm.compute_inputs(datum) for datum in data])
        params, _ = hmm.fit_em(
            params, props, data, inputs=inputs, num_iters=num_iters, verbose=False
        )

    else:
        params, _ = hmm.fit_em(params, props, data, num_iters=num_iters, verbose=False)

    return params


def init_transonly(emissions, probs, ar=False, lags=1):
    """
    Initializes and HMM for transition matrix-only fitting

    Args:
        emissions: the emissions parameters for the HMM
        probs: the initial probabilities for the HMM
        ar: whether or not the model is autoregresssive
        lags: if the model is autoregressive, how many lags to include

    Returns:
        hmm: the appropriate dynamax HMM object with correct hidden states, emission dimension, ar status, etc.
        params: the parameters of the HMM
        props: the properties of the HMM's parameters, which dictate which parameters should be trained
    """
    if ar:
        latdim = emissions.weights.shape[0]
        obsdim = emissions.weights.shape[1]
        hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        params, props = hmm.initialize(
            key=get_key(),
            method="prior",
            initial_probs=probs,
            emission_weights=emissions.weights,
            emission_biases=emissions.biases,
            emission_covariances=emissions.covs,
        )
        props.emissions.weights.trainable = False
        props.emissions.biases.trainable = False
        props.emissions.covs.trainable = False

    else:
        latdim = emissions.means.shape[0]
        obsdim = emissions.means.shape[1]
        hmm = DiagonalGaussianHMM(latdim, obsdim)
        params, props = hmm.initialize(
            key=get_key(),
            method="prior",
            initial_probs=probs,
            emission_means=emissions.means,
            emission_scale_diags=emissions.scale_diags,
        )
        props.emissions.means.trainable = False
        props.emissions.scale_diags.trainable = False

    props.initial.probs.trainable = False

    return hmm, params, props


def logprob_all_models(hmm, params, data, ar=False):
    """
    Calculates the log likelihood of a particular run of data under a specified model

    Args:
        hmm: a dynamax HMM object
        params: the parameters of the HMM
        data: a num_timepoints x num_networks numpy array
        ar: whether or not the model is autoregressive

    Returns:
        an int, the log likelihood of the data under the model
    """
    if ar:
        inputs = hmm.compute_inputs(data)
        return hmm.marginal_log_prob(params, data, inputs=inputs)
    return hmm.marginal_log_prob(params, data)


def loocv(data1, data2, latdim, trans=False, ar=False, lags=1):
    """
    Performs leave-one-out cross-validation by fitting a specified HMM to each dataset
    and classifying test data based on log likelihood under those models
    Args:
        data1: list of num_timepoins x num_networks numpy arrays
        data2: list of num_timepoins x num_networks numpy arrays
        latdim: number of hidden states for the HMM
        trans: whether to fit a transition matrix-only model
        ar: whether to fit an autoregressive model
        lags: if the model is autoregressive, how many lags to include

    Returns:
        The balanced accuracy of the classifier
    """
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    length = min(len(data1), len(data2)) - 1 # need to train each model on the same amount of data,
    # otherwise model trained with more data will have the log likelihood advantage

    if trans:
        emissions, probs = get_saved_params(np.shape(data1[0])[1], latdim, ar=ar, key_string="")
        hmm, base_params1, props = init_transonly(emissions, probs, ar=ar)
        base_params2 = base_params1

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

        base_params2, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data2[:length]))

    correct1 = 0
    params2 = fit_all_models(hmm, base_params2, props, np.array(data2[:length]), ar=ar)
    for i in range(len(data1)):
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        if not trans:
            base_params1, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(temp[:length]))
        params1 = fit_all_models(hmm, base_params1, props, np.array(temp[:length]), ar=ar)
        if logprob_all_models(hmm, params1, data1[i], ar=ar) > logprob_all_models(hmm, params2, data1[i], ar=ar):
            correct1 += 1

    if not trans:
        base_params1, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data1[:length]))

    correct2 = 0
    params1 = fit_all_models(hmm, base_params1, props, np.array(data1[:length]), ar=ar)
    for i in range(len(data2)):
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        if not trans:
            base_params2, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(temp[:length]))
        params2 = fit_all_models(hmm, base_params2, props, np.array(temp[:length]), ar=ar)
        if logprob_all_models(hmm, params1, data2[i], ar=ar) < logprob_all_models(hmm, params2, data2[i], ar=ar):
            correct2 += 1

    return np.average([correct1 / len(data1), correct2 / len(data2)])


def loocv_batch(data1, data2, latdim, trans=False, ar=False, lags=1):
    """
    Performs leave-one-out cross-validation by fitting a specified HMM to each dataset
    and classifying test data based on log likelihood under those models
    in a batched manner that takes advanted of Jax parallelization

    Args:
        data1: list of num_timepoins x num_networks numpy arrays
        data2: list of num_timepoins x num_networks numpy arrays
        latdim: number of hidden states for the HMM
        trans: whether to fit a transition matrix-only model
        ar: whether to fit an autoregressive model
        lags: if the model is autoregressive, how many lags to include

    Returns:
        The balanced accuracy of the classifier
    """
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    length = min(len(data1), len(data2)) - 1

    data1 = jnp.array(data1)
    data2 = jnp.array(data2)
    data1_train = jnp.stack(
        [jnp.concatenate([data1[:i], data1[i + 1:]]) for i in range(len(data1))]
    )
    data2_train = jnp.stack(
        [jnp.concatenate([data2[:i], data2[i + 1:]]) for i in range(len(data2))]
    )

    if trans:
        emissions, probs = get_saved_params(np.shape(data1[0])[1], latdim, ar=ar, key_string="")
        hmm, params, props = init_transonly(emissions, probs, ar)

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

        params, props = hmm.initialize(
            key=get_key(), method="kmeans", emissions=data1[:length, :, :]
        )

    params1 = fit_all_models(hmm, params, props, data1[:length, :, :], ar=ar)
    if not trans:
        params, _ = hmm.initialize(
            key=get_key(), method="kmeans", emissions=data2[:length, :, :]
        )
    params2 = fit_all_models(hmm, params, props, data2[:length, :, :], ar=ar)

    def _fit_fold(train, test, comp_params):
        para = params
        pr = props
        if not trans:
            para, _ = kmeans_init(hmm, train[:length, :, :], get_key(), ar)
        fit_params = fit_all_models(hmm, para, pr, train[:length, :, :], ar)
        return (
                logprob_all_models(hmm, fit_params, test, ar)
                > logprob_all_models(hmm, comp_params, test, ar)
        ).astype(int)

    correct1 = jnp.sum(
        vmap(_fit_fold, in_axes=[0, 0, None])(data1_train, data1, params2)
    )
    correct2 = jnp.sum(
        vmap(_fit_fold, in_axes=[0, 0, None])(data2_train, data2, params1)
    )

    return np.average([correct1 / len(data1), correct2 / len(data2)])


def svmcv(data1, data2):
    """
    Evaluates performance of a linear SVM on functional connectivity matrices using LOO cross-validation

    Args:
        data1: list of num_timepoins x num_networks numpy arrays
        data2: list of num_timepoins x num_networks numpy arrays

    Returns:
        The balanced accuracy of the classifier
    """
    obsdim = np.shape(data1[0])[1]
    fc_1 = []
    fc_2 = []
    for item in data1:
        fc_1.append(np.corrcoef(item.T)[np.triu_indices(obsdim)])
    for item in data2:
        fc_2.append(np.corrcoef(item.T)[np.triu_indices(obsdim)])
    zscored = zscore(np.concatenate((fc_1, fc_2)), axis=0)
    data1 = zscored[: len(data1)].tolist()
    data2 = zscored[len(data1):].tolist()

    length = min(len(data1), len(data2)) - 1
    y = np.concatenate((np.zeros(length), np.ones(length))).tolist()
    correct1 = 0
    correct2 = 0

    for i in range(len(data1)):
        temp1 = data1.copy()
        temp2 = data2.copy()
        temp1.pop(i)
        random.shuffle(temp1)
        random.shuffle(temp2)
        x = np.concatenate((temp1[:length], temp2[:length])).tolist()
        classifier = svm.SVC(kernel="linear")
        classifier.fit(x, y)
        if classifier.predict([data1[i]]) == 0:
            correct1 += 1

    for i in range(len(data2)):
        temp1 = data1.copy()
        temp2 = data2.copy()
        temp2.pop(i)
        random.shuffle(temp1)
        random.shuffle(temp2)
        x = np.concatenate([temp1[:length], temp2[:length]]).tolist()
        classifier = svm.SVC(kernel="linear")
        classifier.fit(x, y)
        if classifier.predict([data2[i]]) == 1:
            correct2 += 1

    return np.average([correct1 / len(data1), correct2 / len(data2)])


def motion_param_svm():
    """
    Evaluates performance of a linear SVM on motion parameters using LOO cross-validation

    Returns:
        The balanced accuracy of the classifier
    """
    x = []
    y = []
    path = os.path.join(root, "Data")
    for filename in os.listdir(os.path.join(path,"data7")):
        if filename[-5] == "t":
            y.append(0)
            data = np.loadtxt(os.path.join(path, "tmasks", (filename[:-5] + filename[-4:])))
            x.append(np.sum(data))
        elif filename[-5] == "r":
            y.append(1)
            data = np.loadtxt(os.path.join(path, "tmasks", (filename[:-5] + filename[-4:])))
            x.append(np.sum(data))
    classifier = svm.SVC(kernel="linear")
    x = list(zscore(x))
    correct1 = 0
    correct2 = 0
    for i in range(len(x)):
        print(i)
        classifier.fit(np.array(x[:i] + x[i+1:]).reshape((-1, 1)), y[:i] + y[i+1:])
        if classifier.predict(np.array(x[i].reshape((-1, 1)))) == y[i]:
            if y[i] == 0:
                correct1 += 1
            else:
                correct2 += 1
    return np.average([correct1 / (len(y) - np.sum(y)), correct2 / np.sum(y)])


def get_transmats(data, latdim, together=False, ar=False, lags=1):
    """
    Fits a transition matrix-only HMM to a dataset and returns the fit transition matrix/matrices

    Args:
        data: a list of num_timepoints x num_networks numpy arrays
        latdim: the number of hidden states in the model
        together: whether or not to fit individual transition matrices for each run
        ar: whether or not the model is autoregressive
        lags: if the model is autoregressive, how many lags to include

    Returns:
        trans: either the fit transition matrix across all the data, or the list of fit transition matrices for each run
    """
    obsdim = np.shape(data[0])[1]
    emissions, probs = get_saved_params(obsdim, latdim, ar=ar)
    hmm, params, props = init_transonly(emissions, probs, ar=ar)

    if together:
        params = fit_all_models(hmm, params, props, jnp.array(data), ar=ar)
        trans = params.transitions.transition_matrix

    else:
        trans = []
        for item in data:
            item_rs = np.reshape(item, [1, np.shape(item)[0], np.shape(item)[1]])
            params = fit_all_models(hmm, params, props, jnp.array(item_rs), ar=ar)
            trans.append(params.transitions.transition_matrix)

    return trans


def permtest(data1, data2, class_func, reps=50, latdim=6, trans=False, ar=False):
    """
    Performs a permutation test to evaluate the significance of a classifier's performance and plots the results

    Args:
        data1: list of num_timepoins x num_networks numpy arrays
        data2: list of num_timepoins x num_networks numpy arrays
        class_func: the classification function to be used
        reps: the number of permutations to perform
        latdim: if using an HMM-based classifier, the number of hidden states
        trans: whether to fit a transition matrix-only model
        ar: whether to fit an autoregressive model

    Returns:
        The average classification accuracy across permutations
    """
    length = math.floor(np.average([len(data1), len(data2)]))
    if class_func == svmcv:
        realacc = svmcv(data1, data2)
    else:
        realacc = class_func(data1, data2, latdim, trans=trans, ar=ar)
    permaccs = []

    for i in range(reps):
        print(i)
        alldatas = data1 + data2
        random.shuffle(alldatas)
        data1 = alldatas[:length]
        data2 = alldatas[length: 2 * length]
        if class_func == svmcv:
            acc = svmcv(data1, data2)
        else:
            acc = class_func(data1, data2, latdim, trans=trans, ar=ar)

        permaccs.append(acc)

    plt.hist(permaccs, bins=50)
    plt.axvline(x=realacc, color="red")
    plt.show()
    return np.average(permaccs)


def get_stats(hmm, params, datas, ar):
    """
    Given a model and data, returns a variety of statistics for the most likely hidden states underlying the data
    given that model
    Args:
        hmm: the hidden markov model
        params: parameters of the model
        datas: list of num_timepoints x num_networks numpy arrays for which to find the hidden states
        ar: whether or not the model is autoregressive

    Returns:
        occs: list of the occupancies in each hidden state across all the data in datas
        dwells: list of the dwell times in each hidden state across all the data in datas
        changes: the number of hidden state transitions per run over all data in datas
    """
    num_states = np.shape(params.transitions.transition_matrix)[0]
    changes = []
    occs = []
    dwells = [[]]
    input = []
    for i in range(num_states - 1):
        dwells.append([])
    for data in datas:
        if ar:
            input = hmm.compute_inputs(data)
        state = hmm.most_likely_states(params, data, inputs=input)
        change = np.nonzero(np.diff(state))
        change = change[0]
        num_change = len(change)
        changes.append(num_change)
        occ = np.histogram(state, bins=num_states, range=(0, num_states))[0]
        occs.append(occ)
        for i in range(num_change):
            if i == 0:
                dwells[state[change[i]]].append(change[0]+1)
            else:
                dwells[state[change[i]]].append(change[i] - change[i - 1])
        dwells[state[-1]].append(len(state) - change[-1] - 1)

    return occs, dwells, changes


def get_state_networks(hmm, params, datas, ar):
    """
    Returns the average networks for each hidden state of the hmm across all the data in datas

    Args:
        hmm: the hidden markov model
        params: parameters of the model
        datas: list of num_timepoints x num_networks numpy arrays for which to find the hidden states
        ar: whether or not the model is autoregressive

    Returns:
        a DataFame of the average activities for each state and network
    """
    num_states = np.shape(params.transitions.transition_matrix)[0]
    num_networks = np.shape(datas[0])[1]
    states = np.arange(num_states)
    activities_dict = {}
    for state in states:
        activities_dict[state] = []
    for data in datas:
        if ar:
            input = hmm.compute_inputs(data)
        ml_states = hmm.most_likely_states(params, data, inputs=input)
        for state in states:
            activities_dict[state] = activities_dict[state] + [data[ml_states == state]]
    activities_list = []
    state_list = []
    networks = (np.arange(num_networks)/num_networks * 2 * math.pi + math.pi/(2*num_networks)).tolist() + [math.pi/(2*num_networks)]
    for state in states:
        state_list += [state]*(num_networks + 1)
        avg_activities = np.average(np.concatenate(activities_dict[state], axis=0), axis=0).tolist()
        activities_list += avg_activities
        activities_list += [avg_activities[0]]
    network_list = networks*num_states
    df = pd.DataFrame({"State": state_list, "Network": network_list, "Activity": activities_list})
    return df


def build_dataframe(directory):
    """
    Builds a dataframe from a directory of classification accuracy files

    Args:
        directory: the directory containing the classification accuracy files

    Returns:
        df: a pandas dataframe containing the classification accuracy data
    """
    df = pd.DataFrame()
    states = np.arange(2, 13)
    for filename in os.listdir(directory):
        data = np.loadtxt(os.path.join(directory, filename))
        network = filename.split("_")[1]
        model_dict = {"full":"Full, HMM", "ar":"Full, ARHMM", "trans": "Trans Only, HMM", "artrans": "Trans Only, ARHMM"}
        category_dict = {"hc":"Healty Control", "psych":"Psychosis"}
        if filename[0] != "b":
            hidden_states = int(filename.split("_")[2])
            #if hidden_states > 8:
                #continue
            model = filename.split("_")[0]
            category = filename.split("_")[3]

            for acc in data:
                df = pd.concat([df,
                    pd.DataFrame({
                        "Confusion Ratio": acc,
                        #"Networks": network,
                        "Category": category_dict[category],
                        "Hidden States": hidden_states,
                        "Model": model_dict[model],
                        #"Motion": motion_dict[motion]
                    }, index=[0])],
                    ignore_index=True,
                )

        elif network == "512":
            model = "Baseline (512-D)"
            for acc in data:
                for state in states:
                    df = df.append(
                        {
                            "Classification Accuracy": acc,
                            "Networks": "7",
                            "Hidden States": state,
                            "Model": model
                        },
                        ignore_index=True,
                    )
                    df = df.append(
                        {
                            "Classification Accuracy": acc,
                            "Networks": "17",
                            "Hidden States": state,
                            "Model": model
                        },
                        ignore_index=True,
                    )

        else:
            model = "zBaseline"
            for acc in data:
                for state in states:
                    df = df.append(
                        {
                            "Classification Accuracy": acc,
                            "Networks": network,
                            "Hidden States": state,
                            "Model": model
                        },
                        ignore_index=True,
                    )

    df.sort_values(by="Model", ascending=True, inplace=True)
    df.loc[df["Model"] == "zBaseline", "Model"] = "Baseline"

    with open(os.path.join(directory, "dataframe"), "wb") as file:
        pickle.dump(df, file)

    return df


def plot_class_acc(dataframe):
    """
    Plots classification accuracy for different models and numbers of hidden states

    Args:
        dataframe: a dataframe with columns "Classification Accuracy", "Model", "Hidden States", and "Networks"

    Returns:
        None
    """
    sns.set_theme()
    colors = [
        [51 / 255, 34 / 255, 136 / 255],
        [136 / 255, 204 / 255, 238 / 255],
        [17 / 255, 119 / 255, 51 / 255],
        [153 / 255, 153 / 255, 51 / 255],
        [204 / 255, 102 / 255, 119 / 255],
        [136 / 255, 34 / 255, 85 / 255],
    ]
    sns.set_palette(sns.color_palette(colors))
    fig = sns.relplot(
        data=dataframe,
        x="Hidden States",
        y="Classification Accuracy",
        hue="Model",
        col="Networks",
        kind="line",
        errorbar="ci"
    ).set_titles("7 Networks", weight="bold", size=14)
    sns.move_legend(fig, "lower right", bbox_to_anchor=(0.815, 0.67))
    fig.legend.set_title(None)
    fig.legend.set(frame_on=True)
    #plt.setp(fig._legend.get_texts(), fontsize=12)

    fig.fig.subplots_adjust(top=0.9)
    plt.ylim([0, 1.05])
    plt.title("17 Networks", weight="bold", fontsize=14)



    fig.tight_layout()
    plt.show()
    #plt.savefig("./results/figs_new/psych_transparent.png", facecolor=(1, 1, 1, 0))

    return None


def plot_occs(hidden_states, data1, data2):
    """
    Make a bar plot comparing the occupancies from two datasets

    Args:
        hidden_states: a list of numbers of hidden states for which to plot occupancy data
        data1: a list of num_timepoins x num_networks numpy arrays
        data2: a list of num_timepoins x num_networks numpy arrays

    Returns:
        dataframe: a dataframe that can be reused to plot occupancy data
    """
    obsdim = np.shape(data1[0])[1]

    states = []
    data = []
    occs = []
    tot_states = []

    for state in hidden_states:
        emissions, probs = get_saved_params(obsdim, state, ar=True, key_string="")
        hmm, params, props = init_transonly(emissions, probs, ar=True)
        params = fit_all_models(hmm, params, props, jnp.array(data1 + data2), ar=True)
        occ1, _, _ = get_stats(hmm, params, data1, ar=True)
        occ2, _, _ = get_stats(hmm, params, data2, ar=True)
        for item in occ1:
            for i in range(0, len(item)):
                occs.append(item[i]/sum(item))
                states.append(i)
                data.append("Uncaffeinated")
                tot_states.append(f"{state} States")
        for item in occ2:
            for i in range(0, len(item)):
                occs.append(item[i]/sum(item))
                states.append(i)
                data.append("Caffeinated")
                tot_states.append(f"{state} States")
    dict = {"Occupancies": occs, "Hidden States": states, "Dataset": data, "Num States": tot_states}
    dataframe = pd.DataFrame(dict)

    sns.set_theme()
    colors = [
        [51 / 255, 34 / 255, 136 / 255],
        [136 / 255, 204 / 255, 238 / 255],
        [17 / 255, 119 / 255, 51 / 255],
        [153 / 255, 153 / 255, 51 / 255],
        [204 / 255, 102 / 255, 119 / 255],
        [136 / 255, 34 / 255, 85 / 255],
    ]
    sns.set_palette(sns.color_palette(colors))
    fig = sns.catplot(data=dataframe, x="Hidden States", y="Occupancies", hue="Dataset", errorbar="ci", col="Num States", kind="bar", sharex=False)
    fig.set_titles("{col_name}")
    fig.legend.set_title(None)
    fig.legend.set(frame_on=True)
    sns.move_legend(fig, "upper right", bbox_to_anchor=(0.9, 0.93))
    plt.subplots_adjust(wspace=0.1)

    plt.show()

    return dataframe


def plot_trans_matrix(mat1, mat2):
    """
    Plots two transition matrices, as well as the element-wise magnitude of their difference
    Args:
        mat1: a numstates x numstates numpy array
        mat2: a numstates x numstates numpy array

    Returns:
        None
    """
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    np.fill_diagonal(mat1, 0)
    np.fill_diagonal(mat2, 0)
    sns.set(font_scale=0.75)
    fig, ax = plt.subplots(ncols=3)
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.345, 0.03, 0.3])
    sns.heatmap(
        mat1, cbar_ax=cbar_ax, ax=ax[0], vmin=0, vmax=0.1, square=True, cmap="viridis"
    )
    sns.heatmap(
        mat2, cbar_ax=cbar_ax, ax=ax[1], vmin=0, vmax=0.1, square=True, cmap="viridis"
    )
    sns.heatmap(
        np.abs(mat2 - mat1),
        cbar_ax=cbar_ax,
        vmin=0,
        vmax=0.1,
        ax=ax[2],
        square=True,
        cmap="viridis",
    )
    ax[0].set_title("Uncaffeinated \nRuns", fontsize=10)
    ax[1].set_title("Caffeinated \nRuns", fontsize=10)
    ax[2].set_title("Absolute \nDifference", fontsize=10)
    plt.savefig(
        "./results/figs/OHBM trans matrices TRANSPARENT", facecolor=(1, 1, 1, 0)
    )
    plt.show()

    return None


def plot_emission_networks(activities_df):
    sns.set(font_scale=0.75)
    sns.set_theme()
    colors = [
        [51 / 255, 34 / 255, 136 / 255],
        [136 / 255, 204 / 255, 238 / 255],
        [17 / 255, 119 / 255, 51 / 255],
        [153 / 255, 153 / 255, 51 / 255],
        [204 / 255, 102 / 255, 119 / 255],
        [136 / 255, 34 / 255, 85 / 255],
    ]
    sns.set_palette(sns.color_palette(colors))


    fig = sns.FacetGrid(activities_df, col="State", hue="State", sharex=True, sharey=False,
                      subplot_kws=dict(projection='polar'), height=4.5, despine=False)
    fig.map(sns.lineplot, "Network", "Activity", sort=False, estimator=None)
    fig.set(xticks=np.arange(7)/7 * 2 * math.pi + math.pi/14)
    fig.set_xticklabels(["Visual", "Somatomotor", "Dorsal\nAttention", "Salience   ", "Limbic", "Control", "Default"])
    fig.set(yticks=np.arange(-0.3, 0.6, 0.3))
    fig.set_yticklabels([-0.3,  0, 0.3])
    fig.set(ylim=(-0.4, 0.4))
    fig.set_axis_labels("")
    fig.set_ylabels("")
    fig.set_titles("State {col_name}", size=14, weight="bold")
    #plt.subplots_adjust(top=0.7)
    plt.xticks(rotation=45)
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def plot_hmm_ll(data, maxstates, folds):
    sns.set_theme()
    colors = [
        [51 / 255, 34 / 255, 136 / 255],
        [136 / 255, 204 / 255, 238 / 255],
        [17 / 255, 119 / 255, 51 / 255],
        [153 / 255, 153 / 255, 51 / 255],
        [204 / 255, 102 / 255, 119 / 255],
        [136 / 255, 34 / 255, 85 / 255],
    ]
    sns.set_palette(sns.color_palette(colors))
    test_ll = np.zeros(maxstates - 1)
    obsdim = np.shape(data[0])[1]
    hiddenstates = np.arange(2, maxstates + 1)
    kf = KFold(n_splits=folds)
    for train, test in kf.split(data):
        for num_states in hiddenstates:
            #hmm = DiagonalGaussianHMM(num_states, obsdim)
            hmm = LinearAutoregressiveHMM(num_states, obsdim, num_lags=1)
            params, props = hmm.initialize(method="kmeans", emissions=np.array(data)[train])
            inputs = []
            for index in train:
                inputs.append(hmm.compute_inputs(np.array(data)[index]))
            params, _ = hmm.fit_em(params, props, np.array(data)[train], num_iters=1000, inputs=np.array(inputs), verbose=False)
            for index in test:
                inputs = hmm.compute_inputs(np.array(data)[index])
                test_ll[num_states - 2] += hmm.marginal_log_prob(params, np.array(data)[index], inputs=inputs)
    sns.lineplot(x=hiddenstates, y=test_ll/len(data))
    plt.xlabel("Hidden States")
    plt.ylabel("Average Log Likelihood")
    plt.title("ARHMM")
    plt.show()


def main():

    directory = os.path.join(root, "results", "fits", "confusion")
    build_dataframe(directory)

    with open(os.path.join(directory, "dataframe"), "rb") as file:
        dataframe = pickle.load(file)

    sns.set_theme()
    colors = [
        [51 / 255, 34 / 255, 136 / 255],
        [136 / 255, 204 / 255, 238 / 255],
        [17 / 255, 119 / 255, 51 / 255],
        [153 / 255, 153 / 255, 51 / 255],
        [204 / 255, 102 / 255, 119 / 255],
        [136 / 255, 34 / 255, 85 / 255],
    ]
    sns.set_palette(sns.color_palette(colors))
    fig = sns.catplot(data=dataframe, x="Hidden States", y="Confusion Ratio", hue="Category", errorbar="ci", col="Model", kind="bar", sharex=False)
    fig.set_titles("{col_name}")
    fig.legend.set_title(None)
    fig.legend.set(frame_on=True)
    fig.refline(y=1, color=colors[4])
    sns.move_legend(fig, "upper right", bbox_to_anchor=(0.89, 0.93))
    plt.setp(fig._legend.get_texts(), fontsize=12)
    plt.subplots_adjust(wspace=0.1)

    plt.show()

    quit()


    plot_class_acc(dataframe)
    quit()
    tues, thurs = import_tuesthurs(7)
    loocv_batch(tues, thurs, 5)

    quit()

    with (open(os.path.join(root, "results", "fits", "MyConnectome", "dataframe"), "rb")) as file:
        data = pickle.load(file)

    data = data.rename({"Classification Accuracy":"Classification_Accuracy", "Hidden States":"Hidden_States"}, axis=1)
    #plot_class_acc(data)
    model = smf.ols("Classification_Accuracy ~ Model + Hidden_States + Networks", data)
    result = model.fit()
    print(result.summary())

    model2 = smf.ols("Classification_Accuracy ~ Hidden_States + Networks", data)
    result2 = model2.fit()
    #compare_lr_test


    asfgw4352t3w2341


    data = import_all(7)
    plot_hmm_ll(data, 12, 5)
    quit()

    data = import_all(7)
    emissions, probs = get_saved_params(7, 5, ar=True, key_string="")
    hmm, params, props = init_transonly(emissions, probs, ar=True)
    params = fit_all_models(hmm, params, props, jnp.array(data), ar=True)


    df = get_state_networks(hmm, params, data, ar=True)
    plot_emission_networks(df)

if __name__ == "__main__":
    main()
