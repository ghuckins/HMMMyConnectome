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

    params, props = hmm.initialize(
        key=get_key(), method="kmeans", emissions=np.array(data)
    )

    if ar:
        params, ll = hmm.fit_em(
            params, props, np.array(data), inputs=np.array(inputs), num_iters=1000
        )
    else:
        params, ll = hmm.fit_em(params, props, np.array(data), num_iters=1000)

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
        emissions, probs = get_saved_params(
            np.shape(data1[0])[1], latdim, ar=ar, key_string=""
        )
        hmm, base_params1, props = init_transonly(emissions, probs, ar=ar)
        base_params2 = base_params1

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

        base_params2, props = hmm.initialize(
            key=get_key(), method="kmeans", emissions=np.array(data2[:length])
        )

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
        base_params1, _ = hmm.initialize(
            key=get_key(), method="kmeans", emissions=np.array(data1[:length])
        )

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


def getstats(model, params, datas, num_states, ar):
    """
    THIS FUNCTION IS UNREVIEWED
    """
    changes = []
    occs = []
    dwells = [[]]
    input = []
    for i in range(num_states - 1):
        dwells.append([])
    reps = 1
    for rep in np.arange(reps):
        for data in datas:
            if ar:
                input = model.compute_inputs(data)
            state = model.most_likely_states(params, data, inputs=input)
            change = np.nonzero(np.diff(state))
            change = change[0]
            num_change = len(change)
            changes.append(num_change)
            occ = np.histogram(state, bins=num_states, range=(0, num_states))[0]
            occs.append(occ)
            for i in range(num_change):
                if i == 0:
                    dwells[state[change[0]]].append(change[0])
                else:
                    dwells[state[change[i]]].append(change[i] - change[i - 1])
        avgchanges = np.mean(changes)
        stdchanges = np.std(changes)
        avgoccs = np.mean(occs, axis=0)
        stdoccs = np.std(occs, axis=0)
        avgdwells = []
        stddwells = []
        for item in dwells:
            avgdwells.append(np.mean(item))
            stddwells.append(np.std(item))
        return occs  # avgoccs, avgdwells, avgchanges, stdoccs, stddwells, stdchanges


def build_dataframe(directory):
    """
    NOT VERIFIED! UPDATE AND CHECK BASED ON NAMING SCHEME I END UP USING
    Builds a dataframe from a directory of classification accuracy files

    Args:
        directory: the directory containing the classification accuracy files

    Returns:
        df: a pandas dataframe containing the classification accuracy data
    """
    df = pd.DataFrame()
    for filename in os.listdir(directory):
        data = np.loadtxt(os.path.join(directory, filename))
        hidden_states = int(filename.split("_")[-1])
        method = filename.split("_")[-2]
        if filename.startswith("b"):
            model = "full"
        else:
            model = filename.split("_")[0]

        for acc in data:
            df = df.append(
                {
                    "Classification Accuracy": acc,
                    "Model": model,
                    "Hidden States": hidden_states,
                    "Method": method
                },
                ignore_index=True,
            )
    return df


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
    #dataframe = dataframe[dataframe["Method"] == "batch"]
    sns.set_palette(sns.color_palette(colors))
    fig = sns.relplot(
        data=dataframe,
        x="Hidden States",
        y="Classification Accuracy",
        hue="Method",
        col="Model",
        kind="line",
        errorbar="ci",
    ) #.set_titles("7 Networks", weight="bold", size=14)
    sns.move_legend(fig, "upper right", bbox_to_anchor=(0.817, 0.93))
    fig.legend.set_title(None)
    fig.legend.set(frame_on=True)

    fig.fig.subplots_adjust(top=0.9)
    plt.ylim([0.3, 1])
    #plt.title("17 Networks", weight="bold", fontsize=14)

    fig.tight_layout()
    plt.show()

    return None

def main():
    dir = os.path.join(data_root, "HCP")
    for file in os.listdir(dir):
        data = np.load(os.path.join(dir,file))
        get_network_activity(data, 7, hcp=True)
        quit()
    reps = 5
    states = np.arange(2,7)
    for state in states:
        total = 0
        for rep in range(reps):
            total += loocv_batch(tues, thurs, state, trans=False, ar=True)
        print(total / reps)

def still_to_edit():
    hm_fc = {}
    hm_fc["t"] = []
    hm_fc["r"] = []
    nohm_fc = {}
    nohm_fc["t"] = []
    nohm_fc["r"] = []

    path1 = os.path.join(root, "results", "data7_split")
    path2 = os.path.join(root, "results", "data7")
    thresh = 100

    for filename in os.listdir(path2):
        if filename[6] != "t" and filename[6] != "r":
            continue

        data = np.loadtxt(os.path.join(path2, filename))
        num = 0
        hm_pos = 0
        while os.path.exists(os.path.join(path1, filename[:7] + f"_{num}.txt")):
            data1 = np.loadtxt(os.path.join(path1, filename[:7] + f"_{num}.txt"))
            counter = 1
            while counter <= len(data1) / thresh:
                hm_pos += 1
                nohmdata = data1[(counter - 1) * thresh: (counter) * thresh]
                hmdata = data[(hm_pos - 1) * thresh: (hm_pos) * thresh]
                nohm_fc[filename[6]].append(np.corrcoef(nohmdata.T)[np.triu_indices(7)])
                hm_fc[filename[6]].append(np.corrcoef(hmdata.T)[np.triu_indices(7)])
                counter += 1
            num += 1

    bl_motion = []
    bl_nomotion = []
    reps = 10
    for rep in range(reps):
        bl_motion.append(svmcv(hm_fc["t"], hm_fc["r"]))
        bl_nomotion.append(svmcv(nohm_fc["t"], nohm_fc["r"]))
        print(bl_motion[-1])
        print(bl_nomotion[-1])
    np.savetxt(
        os.path.join(root, "results", "fits", "hmm7_split", "bl_motion"), bl_motion
    )
    np.savetxt(
        os.path.join(root, "results", "fits", "hmm7_split", "bl_nomotion"), bl_nomotion
    )
    quit()

    tues, thurs = import_tuesthurs(17)
    all_data = tues.copy()
    all_data.extend(thurs)

    emissions, probs = get_saved_params(tues, 4, True)
    hmm, params, props = init_transonly(emissions, probs, True)
    params = fit_all_models(hmm, params, props, np.array(all_data), True)

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

    tuesoccs = np.array(getstats(hmm, params, tues, 4, True)) / 518
    thursoccs = getstats(hmm, params, thurs, 4, True)
    tuesoccs = tuesoccs.reshape(-1, 1)
    tuesstates = np.array([[0, 1, 2, 3] * 30]).T
    thursoccs = np.array(getstats(hmm, params, thurs, 4, True)) / 518
    thursoccs = thursoccs.reshape(-1, 1)
    alloccs = np.concatenate((tuesoccs, thursoccs), axis=0)
    thursstates = np.array([[0, 1, 2, 3] * 23]).T
    allstates = np.concatenate((tuesstates, thursstates), axis=0)
    labels = ["Uncaffeinated"] * 120
    labels.extend(["Caffeinated"] * 92)
    occsdict = {}
    occsdict["Occupancy"] = alloccs.flatten()
    occsdict["State"] = allstates.flatten()
    occsdict["Label"] = labels
    print(len(occsdict["Occupancy"]))
    print(len(occsdict["State"]))
    print(len(occsdict["Label"]))
    occsdf = pd.DataFrame(data=occsdict)
    ax = sns.barplot(
        occsdf, x="State", y="Occupancy", hue="Label", estimator="mean", errorbar="ci"
    )
    # fig.legend.set_title(None)
    ax.legend(title=None)
    plt.savefig("./results/figs/OHBM occupancies TRANSPARENT", facecolor=(1, 1, 1, 0))
    plt.show()

if __name__ == "__main__":
    main()
