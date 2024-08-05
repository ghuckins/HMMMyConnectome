import numpy as np
import os
import pandas as pd
from hmm import get_params, get_saved_params, init_transonly, get_key, fit_all_models, logprob_all_models
import jax.numpy as jnp
from jax import vmap
from scipy.stats import zscore
from sklearn import svm
from k_means import kmeans_init
import sys
import random

import warnings
warnings.filterwarnings("ignore")

from src.dynamax.hidden_markov_model.models.gaussian_hmm import DiagonalGaussianHMM
from src.dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM

root = "/oak/stanford/groups/russpold/users/ghuckins/MyConnectome_Current"
data_root = "/Users/gracehuckins/Documents/Research Data"

def import_all(num_networks):
    data_path = os.path.join(data_root, "psychosis")
    path = os.path.join(root, "data", f"psychosis{num_networks}")
    good_subjects = np.loadtxt(os.path.join(data_root, "psychosis", "usable.txt"), dtype=str)
    for filename in os.listdir(data_path):
        if filename.endswith("timeseries.tsv"):
            file_key = filename[4:12]
            if (file_key in good_subjects
                    and not os.path.exists(os.path.join(path, filename[:35]+"_999.txt"))
                    and not os.path.exists(os.path.join(path, filename[:35]+"_888.txt"))):
                import_raw(filename, num_networks)
    data = []
    for filename in os.listdir(path):
        data.append(np.loadtxt(os.path.join(path, filename)))
    return data

def import_psych(num_networks):
    psych_data = {}
    hc_data = {}
    path = os.path.join(root, "data", f"psychosis{num_networks}")
    good_subjects = np.loadtxt(os.path.join(data_root, "psychosis", "usable.txt"), dtype=str)
    for filename in os.listdir(path):
        file_key = filename[4:12]
        if file_key in good_subjects:
            if filename[-5] == "8":
                if file_key in psych_data.keys():
                    psych_data[file_key].append(np.loadtxt(os.path.join(path, filename)))
                else:
                    psych_data[file_key] = [np.loadtxt(os.path.join(path, filename))]
            elif filename[-5] == "9":
                if file_key in hc_data.keys():
                    hc_data[file_key].append(np.loadtxt(os.path.join(path, filename)))
                else:
                    hc_data[file_key] = [np.loadtxt(os.path.join(path, filename))]
    return psych_data, hc_data

def import_raw(filename, num_networks):
    path = os.path.join(data_root, "psychosis")
    outliers = pd.read_table(os.path.join(path, "outliers", filename[:35] + "_outliers.tsv")).values
    if outliers[0] == 1 or outliers[-1] == 1 or np.sum(outliers) > 0.25*len(outliers):
        return None

    metadata = pd.read_table(os.path.join(path, "participants.tsv"))
    metadata = metadata.set_index("participant_id")

    savepath = os.path.join(root, "data", f"psychosis{num_networks}")
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    psychosis = metadata.loc[filename[:12]]["is_this_subject_a_patient"]
    data = pd.read_table(os.path.join(path, filename))
    data.dropna(axis=1, inplace=True)
    numpy_data = np.array(data)
    numpy_data = interpolate(numpy_data, outliers)
    data = pd.DataFrame(numpy_data, columns=data.columns)

    activities = get_network_activity(data, num_networks)
    activities = zscore(activities)

    if np.sum(np.isnan(activities)) == 0 and np.shape(activities)[0] == 415:
        np.savetxt(os.path.join(savepath, filename[:35] + f"_{int(psychosis)}.txt"), activities)

    return None

def get_network_activity(data, num_networks):

    parcellation = pd.read_table(os.path.join(data_root, "psychosis", "networks.tsv"))
    net_header = "network_label"
    if num_networks == 17:
        net_header += "_17network"
    networks = (pd.unique(parcellation[net_header]))
    networks.sort()

    activities = []
    for network in networks:
        regions = parcellation[(parcellation[net_header] == network)]["label"].values
        network_data = data[data.columns.intersection(regions)]
        avg_data = np.mean(network_data.values, axis=1).reshape((-1, 1))
        activities.append(avg_data)

    return np.concatenate(activities, axis=1)


def interpolate(timeseries, outliers):
    loc = 0
    while loc < len(outliers):
        if outliers[loc] == 0:
            loc += 1
        else:
            start = loc - 1
            while outliers[loc] == 1:
                loc += 1
            to_interpolate = np.linspace(timeseries[start, :], timeseries[start + 1, :], loc - start - 1)
            timeseries = np.concatenate((timeseries[:start + 1, :], to_interpolate, timeseries[start + 1:, :]))
    return timeseries


def loocv_batch_loso(data1, data2, latdim, trans=False, ar=False, lags=1):
    obsdim = np.shape(list(data1.values())[0])[2]

    keys_1 = list(data1.keys())
    keys_2 = list(data2.keys())
    random.shuffle(keys_1)
    random.shuffle(keys_2)
    length = (min(len(keys_1), len(keys_2)) - 1)*4

    data1_train = []
    data1_test = []

    for key in keys_1:
        data1_test.extend([data1[key][i] for i in range(4)])
        templist = []
        keys_temp = keys_1.copy()
        random.shuffle(keys_temp)
        for k in keys_temp:
            if k != key:
                templist.extend(data1[k])
        templist = jnp.stack(templist)
        for j in range(4):
            data1_train.append(templist)

    data1_test = jnp.stack(data1_test)
    data1_train = jnp.stack(data1_train)

    data2_train = []
    data2_test = []

    for key in keys_2:
        data2_test.extend([data2[key][i] for i in range(4)])
        templist = []
        keys_temp = keys_2.copy()
        random.shuffle(keys_temp)
        for k in keys_temp:
            if k != key:
                templist.extend(data2[k])
        templist = jnp.stack(templist)
        for j in range(4):
            data2_train.append(templist)

    data2_test = jnp.stack(data1_test)
    data2_train = jnp.stack(data1_train)

    data1_values = jnp.array([x for i in data1.values() for x in i])
    data2_values = jnp.array([x for i in data2.values() for x in i])

    if trans:
        emissions, probs = get_saved_params(obsdim, latdim, ar=ar, key_string="psych")
        hmm, params, props = init_transonly(emissions, probs, ar)

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

        params, props = hmm.initialize(
            key=get_key(), method="kmeans", emissions=data1_values[:length, :, :]
        )

    params1 = fit_all_models(hmm, params, props, data1_values[:length, :, :], ar=ar)

    if not trans:
        params, _ = hmm.initialize(
            key=get_key(), method="kmeans", emissions=data2_values[:length, :, :]
        )
    params2 = fit_all_models(hmm, params, props, data2_values[:length, :, :], ar=ar)

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
        vmap(_fit_fold, in_axes=[0, 0, None])(data1_train, data1_test, params2)
    )
    correct2 = jnp.sum(
        vmap(_fit_fold, in_axes=[0, 0, None])(data2_train, data2_test, params1)
    )

    return np.average([correct1 / len(data1_values), correct2 / len(data2_values)])

def loocv_batch_loro(data1, data2, latdim, trans=False, ar=False, lags=1):

    data1 = [x for i in data1.values() for x in i]
    data2 = [x for i in data2.values() for x in i]

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
        emissions, probs = get_saved_params(np.shape(data1[0])[1], latdim, ar=ar, key_string="psych")
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


def fingerprinting_confusion(data, latdim, num_subjs, trans=False, ar=False, lags=1):

    obsdim = np.shape(list(data.values())[0])[2]

    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    params = {}

    confusion = pd.DataFrame(data=np.zeros((num_subjs,num_subjs)), index=keys, columns=keys)

    if trans:
        emissions, probs = get_saved_params(obsdim, latdim, ar=ar, key_string="psych")
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

def svmloso(data1, data2):

    obsdim = np.shape(data1[0])[1]
    data1 = [x for i in data1.values() for x in i]
    data2 = [x for i in data2.values() for x in i]
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

    for i in range(len(data1)/4):
        temp1 = data1.copy()
        temp2 = data2.copy()
        temp1.pop(i*4)
        temp1.pop(i*4 + 1)
        temp1.pop(i*4 + 2)
        temp1.pop(i*4 + 3)
        random.shuffle(temp1)
        random.shuffle(temp2)
        x = np.concatenate((temp1[:length], temp2[:length])).tolist()
        classifier = svm.SVC(kernel="linear")
        classifier.fit(x, y)
        if classifier.predict([data1[i*4]]) == 0:
            correct1 += 1
        if classifier.predict([data1[i*4 + 1]]) == 0:
            correct1 += 1
        if classifier.predict([data1[i*4 + 2]]) == 0:
            correct1 += 1
        if classifier.predict([data1[i*4 + 3]]) == 0:
            correct1 += 1

    for i in range(len(data2)/4):
        temp1 = data1.copy()
        temp2 = data2.copy()
        temp2.pop(i*4)
        temp2.pop(i*4 + 1)
        temp2.pop(i*4 + 2)
        temp2.pop(i*4 + 3)
        random.shuffle(temp1)
        random.shuffle(temp2)
        x = np.concatenate([temp1[:length], temp2[:length]]).tolist()
        classifier = svm.SVC(kernel="linear")
        classifier.fit(x, y)
        if classifier.predict([data2[i*4]]) == 1:
            correct2 += 1
        if classifier.predict([data2[i*4 + 1]]) == 1:
            correct2 += 1
        if classifier.predict([data2[i*4 + 2]]) == 1:
            correct2 += 1
        if classifier.predict([data2[i*4 + 3]]) == 1:
            correct2 += 1

    return np.average([correct1 / len(data1), correct2 / len(data2)])


def svmloro(data1, data2):

    obsdim = np.shape(data1[0])[1]
    data1 = [x for i in data1.values() for x in i]
    data2 = [x for i in data2.values() for x in i]
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

    return correct/(num_subjs*4), confusion


def baseline_fingerprint(data, num_subjs):
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

    data7 = import_all(7)
    data17 = import_all(17)
    for states in range(2,13):
        get_params(data7, states, key_string="psych")
        get_params(data7, states, ar=True, key_string="psych")
        get_params(data17, states, key_string="psych")
        get_params(data17, states, ar=True, key_string="psych")

    #still need to get all other params



if __name__ == "__main__":
    main()