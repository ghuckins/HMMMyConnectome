import numpy as np
import os
import pandas as pd
import random
from scipy.stats import zscore
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
import jax.random as jr
import jax.numpy as jnp
import math
import seaborn as sns
from jax import vmap
from k_means import kmeans_init

import warnings
warnings.filterwarnings("ignore")

from src.dynamax.hidden_markov_model.models.gaussian_hmm import DiagonalGaussianHMM
from src.dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM

root = "/Users/gracehuckins/PycharmProjects/HMMMyConnectome"
data_root = "/Users/gracehuckins/Documents/Research Data"


def import_tuesthurs(num_networks, split=False, heldout=False):
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
    elif heldout:
        path = os.path.join(root, "data", f"data{num_networks}_heldout")
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


def import_all(num_networks):
    """
    Imports all the MyConnectome data as a single list of numpy arrays

    Args:
        num_networks: int (7 or 17) indicating whether to import 7- or 17-network parcellation

    Returns:
        data: list of numpy arrays, each array is one recording
    """
    path1 = os.path.join(root, "data", f"data{num_networks}_heldout")
    path2 = os.path.join(root, "data", f"data{num_networks}")
    data = []
    for file_name in os.listdir(path1):
        data.append(np.loadtxt(os.path.join(path1, file_name)))
    for file_name in os.listdir(path2):
        data.append(np.loadtxt(os.path.join(path2, file_name)))
    return data


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
        params, _ = hmm.fit_em(params, props, np.array(data), inputs=np.array(inputs), num_iters=1000)
    else:
        params, _ = hmm.fit_em(params, props, np.array(data), num_iters=1000)

    np.save(os.path.join(path, f"probs{latdim}"), params.initial.probs)

    with open(os.path.join(path, f"emissions{latdim}"), "wb") as file:
        pickle.dump(params.emissions, file)

    return None


def get_key():
    """
    Returns a random PRNG key in order to randomize model fitting and other stochastic Jax-based operations

    Returns: A PRNG key usable by Jax functions
    """
    return jr.PRNGKey(random.randint(0, 10000))


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
    Initializes an HMM for transition matrix-only fitting

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


def oos_test(data1, data2, data1_oos, data2_oos, latdim, trans=False, ar=False, lags=1):
    """
    Tests HMM-based classification on held-out data by training two HMMs on the in-sample data
    and labeling each held-out data point according the model that assigns it the higher log likelihood
    Args:
        data1: list of num_timepoins x num_networks numpy arrays (in-sample data, label 1)
        data2: list of num_timepoins x num_networks numpy arrays (in-sample data, label 2)
        data1_oos: list of num_timepoins x num_networks numpy arrays (out-of-sample data, label 1)
        data2_oos: list of num_timepoins x num_networks numpy arrays (out-of-sample data, label 2)
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
    length = min(len(data1), len(data2)) # need to train each model on the same amount of data,
    # otherwise model trained with more data will have the log likelihood advantage
    data1 = data1[:length]
    data2 = data2[:length]

    if trans:
        emissions, probs = get_saved_params(obsdim, latdim, ar=ar, key_string="heldout")
        hmm, base_params, props = init_transonly(emissions, probs, ar=ar)
        base_params2 = base_params

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

        base_params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data1))
        base_params2, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data2))

    params1 = fit_all_models(hmm, base_params, props, np.array(data1), ar=ar)
    params2 = fit_all_models(hmm, base_params2, props, np.array(data2), ar=ar)

    correct1 = 0
    for item in data1_oos:
        if logprob_all_models(hmm, params1, item, ar=ar) > logprob_all_models(hmm, params2, item, ar=ar):
            correct1 += 1

    correct2 = 0
    for item in data2_oos:
        if logprob_all_models(hmm, params1, item, ar=ar) < logprob_all_models(hmm, params2, item, ar=ar):
            correct2 += 1
    return np.average([correct1 / len(data1_oos), correct2 / len(data2_oos)])


def svmcv(data1, data2, data1_oos, data2_oos):
    """
    Tests a linear SVM on functional connectivity matrices calculated from out-of-sample data

    Args:
        data1: list of num_timepoins x num_networks numpy arrays (in-sample data, label 1)
        data2: list of num_timepoins x num_networks numpy arrays (in-sample data, label 2)
        data1_oos: list of num_timepoins x num_networks numpy arrays (out-of-sample data, label 1)
        data2_oos: list of num_timepoins x num_networks numpy arrays (out-of-sample data, label 2)

    Returns:
        The balanced accuracy of the classifier
    """
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    length = min(len(data1), len(data2))
    data1 = data1[:length]
    data2 = data2[:length]
    fc_1 = []
    fc_2 = []
    for item in data1:
        fc_1.append(np.corrcoef(item.T)[np.triu_indices(obsdim)])
    for item in data2:
        fc_2.append(np.corrcoef(item.T)[np.triu_indices(obsdim)])
    x = zscore(np.concatenate((fc_1, fc_2)), axis=0).tolist()
    y = np.concatenate((np.zeros(length), np.ones(length))).tolist()
    classifier = svm.SVC(kernel="linear")
    classifier.fit(x, y)

    fc_1_oos = []
    fc_2_oos = []
    for item in data1_oos:
        fc_1_oos.append(np.corrcoef(item.T)[np.triu_indices(obsdim)])
    for item in data2_oos:
        fc_2_oos.append(np.corrcoef(item.T)[np.triu_indices(obsdim)])
    x_oos = zscore(np.concatenate((fc_1_oos, fc_2_oos)), axis=0).tolist()

    y_oos = classifier.predict(x_oos)
    length = len(data1_oos)

    return np.average([(length - np.sum(y_oos[:length])) / length, np.sum(y_oos[length:]) / len(data2_oos)])


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
        if filename[0]!= "b":
            data = np.loadtxt(os.path.join(directory, filename))
            hidden_states = int(filename.split("_")[2])
            networks = filename.split("_")[1]
            model = filename.split("_")[0]
            motion = filename.split("_")[3]

            model_dict = {"full": "Gaussian, Full", "trans": "Gaussian, Trans Only", "ar": "Autoregressive, Full", "artrans": "Autoregressive, Trans Only"}
            motion_dict = {"motion": "Head Motion Interpolated", "nomotion": "Head Motion Censored"}
            for acc in data:
                df = df.append(
                    {
                        "Classification Accuracy": acc,
                        "Networks": networks,
                        "Hidden States": hidden_states,
                        "Model": model_dict[model],
                        "Motion": motion_dict[motion]
                    },
                    ignore_index=True,
                )
                """
        else:
            data = np.loadtxt(os.path.join(directory, filename))
            networks = filename.split("_")[1]

            for acc in data:
                for state in range(2, 9):
                    if networks != "512":
                        df = df.append(
                            {
                                "Classification Accuracy": acc,
                                "Networks": networks,
                                "Hidden States": state,
                                "Model": "Baseline"
                            },
                            ignore_index=True,
                        )
                    else:
                        df = df.append(
                            {
                                "Classification Accuracy": acc,
                                "Networks": "7",
                                "Hidden States": state,
                                "Model": "Baseline (512-D)"
                            },
                            ignore_index=True,
                        )
                        df = df.append(
                            {
                                "Classification Accuracy": acc,
                                "Networks": "17",
                                "Hidden States": state,
                                "Model": "Baseline (512-D)"
                            },
                            ignore_index=True,
                        ) """
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
        hue="Motion",
        col="Model",
        kind="line",
        errorbar="ci"
    ).set_titles("Autoregressive", weight="bold", size=14)
    sns.move_legend(fig, "lower right", bbox_to_anchor=(0.78, 0.15))
    fig.legend.set_title(None)
    fig.legend.set(frame_on=True)

    fig.fig.subplots_adjust(top=0.9)
    plt.ylim([0, 1.05])
    plt.xticks(range(2, 13))

    plt.title("Gaussian", weight="bold", fontsize=14)

    fig.tight_layout()
    plt.show()

    return None
def main():
    max_states = 12
    states = range(2, max_states + 1)
    num_networks = 17

    data = import_all(num_networks)
    for state in states:
        get_params(data, state, key_string="heldout")
        get_params(data, state, ar=True, key_string="heldout")

    savepath = os.path.join(root, "results", "fits", "MyConnectome_OOS")

    tues, thurs = import_tuesthurs(num_networks)
    tues_oos, thurs_oos = import_tuesthurs(num_networks, heldout=True)

    num_reps = 25

    svm_acc = []
    for rep in range(num_reps):
         svm_acc.append(svmcv(tues, thurs, tues_oos, thurs_oos))
    with open(os.path.join(savepath, f"baseline{num_networks}_OOS"), "ab") as file:
        np.savetxt(file, svm_acc)

    for state in states:
        print(state)
        full_acc = []
        trans_acc = []
        ar_acc = []
        artrans_acc = []
        for rep in range(num_reps):
            full_acc.append(oos_test(tues, thurs, tues_oos, thurs_oos, state))
            trans_acc.append(oos_test(tues, thurs, tues_oos, thurs_oos, state, trans=True))
            ar_acc.append(oos_test(tues, thurs, tues_oos, thurs_oos, state, ar=True))
            artrans_acc.append(oos_test(tues, thurs, tues_oos, thurs_oos, state, trans=True, ar=True))
        with open(os.path.join(savepath, f"full_{num_networks}_{state}"), "ab") as file:
            np.savetxt(file, full_acc)
        with open(os.path.join(savepath, f"trans_{num_networks}_{state}"), "ab") as file:
            np.savetxt(file, trans_acc)
        with open(os.path.join(savepath, f"ar_{num_networks}_{state}"), "ab") as file:
            np.savetxt(file, ar_acc)
        with open(os.path.join(savepath, f"artrans_{num_networks}_{state}"), "ab") as file:
            np.savetxt(file, artrans_acc)


if __name__ == "__main__":
    main()
