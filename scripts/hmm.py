import numpy as np
import ssm
import os
import pandas as pd
import random
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn import svm
import math
import matplotlib.pyplot as plt
import pickle
import time
import jax.random as jr

from dynamax.hidden_markov_model import DiagonalGaussianHMM

root = "/Users/gracehuckins/Documents/HMMMyConnectome"

def import_all(num_networks):
    path = os.path.join(root,"results",f"data{num_networks}")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    data = []
    for file_name in os.listdir(path):
        data.append(np.loadtxt(os.path.join(path,file_name)))
    return data

def import_tuesthurs(num_networks):
    path = os.path.join(root,"results",f"data{num_networks}")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    tues_data = []
    thurs_data = []
    for filename in os.listdir(path):
        if filename.endswith('t.txt'):
            tues_data.append(np.loadtxt(os.path.join(path,filename)))
        if filename.endswith('r.txt'):
            thurs_data.append(np.loadtxt(os.path.join(path,filename)))
    return tues_data, thurs_data

def import_raw(num_networks):
    print("importing")
    path = os.path.join(root,"data")
    savepath = os.path.join(root,"results",f"data{num_networks}")

    metadata = pd.read_table(os.path.join(path, "trackingdata.txt")) #download trackingdata_goodscans
    metadata = metadata.set_index("subcode")

    for filename in os.listdir(path):
        if filename.startswith("sub"):
            day = metadata.loc[filename.replace(".txt","")]["day_of_week"]
            raw_data = np.loadtxt(os.path.join(path,filename))
            raw_data = gsr(raw_data)
            activities = zscore(getNetworkActivity(raw_data, num_networks), axis=0)

            if day == "2":
                np.savetxt(os.path.join(savepath, filename.replace(".txt","t.txt")), activities)
            if day == "4":
                np.savetxt(os.path.join(savepath, filename.replace(".txt","r.txt")), activities)
            else:
                np.savetxt(os.path.join(savepath, filename), activities)

#instead—numpy structured array to label columns
def getNetworkActivity(data, num_networks):
    parcellation = f"{num_networks}networks"
    parcels = pd.read_table(os.path.join(root,"data/parcel_data.txt"))[parcellation]
    roughnetworks = pd.unique(parcels)
    activities = []
    #assert that length of parcels and #features in data are the same
    for network in roughnetworks:
        if network.lower().startswith(parcellation):
            netactivity = np.average(
                data[:, (parcels == network)], axis=1
            ).reshape((-1, 1))
            activities.append(netactivity)
    return np.concatenate(activities, axis=1)

def gsr(data):
    gsignal = np.average(data, axis=1)
    gsignal = np.reshape(gsignal, (-1, 1))
    ginverse = np.linalg.inv(gsignal.T @ gsignal) @ gsignal.T
    return data - ginverse @ data

def get_transmats(data,latdim):
    obsdim = np.shape(data[0])[1]
    trans = []

    if not os.path.exists(os.path.join(root,"results",f"jaxmodel{obsdim}",f"means{latdim}")):
        get_params_jax(obsdim,latdim)

    means = np.loadtxt(os.path.join(root,"results",f"jaxmodel{obsdim}",f"means{latdim}"))
    scale_diags = np.loadtxt(os.path.join(root,"results",f"jaxmodel{obsdim}",f"scale_diags{latdim}"))
    probs = np.loadtxt(os.path.join(root,"results",f"jaxmodel{obsdim}",f"probs{latdim}"))

    hmm = DiagonalGaussianHMM(latdim, obsdim)
    params, props = hmm.initialize(initial_probs=probs, emission_means=means, emission_scale_diags=scale_diags)
    props.emissions.means.trainable = False
    props.emissions.scale_diags.trainable = False
    props.initial.probs.trainable = False

    for item in data:
        item_params, _ = hmm.fit_em(params, props, item, num_iters=100, verbose=False)
        trans.append(item_params.transitions.transition_matrix.reshape(-1))

    return trans

def get_params_jax(obsdim, latdim):
    '''
    calculates and saves the fit initial state and observation model for a given hmm on the full data

    Args:
        obsdim: the model output dimension—the number of networks being fit
        latdim: the model latent dimension

    Returns:
        nothing
    '''
    path = os.path.join(root,"results",f"jaxmodel{obsdim}")
    if not os.path.exists(path):
        os.mkdir(path)
    hmm = DiagonalGaussianHMM(latdim, obsdim)
    data = import_all(obsdim)
    params, props = hmm.initialize()
    params, _ = hmm.fit_em(params, props, np.array(data), num_iters=1000)
    np.savetxt(os.path.join(path,f"probs{latdim}"), params.initial.probs)
    np.savetxt(os.path.join(path,f"means{latdim}"), params.emissions.means)
    np.savetxt(os.path.join(path,f"scale_diags{latdim}"), params.emissions.scale_diags)

def loocv(data1, data2, latdim1, latdim2):
    '''
    Performs LOO cross-validation by fitting separate HMMs to each dataset and classifying the left-out data
    based on which HMM assigns it a higher log likelihood

    Args:
        data1, data2: datasets to be classified
        latdim1: latent dimensionality for data1
        latdim2: latent dimensionality for data2

    Returns: Balanced accuracy of the classifier

    '''
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    assert np.shape(data2[0])[1] == obsdim, "Datasets must have the same number of features"
    length = min(len(data1), len(data2)) - 1
    correct1 = 0
    hmm1 = DiagonalGaussianHMM(latdim1, obsdim)
    hmm2 = DiagonalGaussianHMM(latdim2, obsdim)
    base_params1, props1 = hmm1.initialize()
    base_params2, props2 = hmm2.initialize()
    params2, _ = hmm2.fit_em(base_params2, props2, np.array(data2[:length]), num_iters=100)
    for i in range(len(data1)):
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        params1, _ = hmm1.fit_em(base_params1, props1, np.array(temp[:length]), num_iters=100)
        if hmm1.marginal_log_prob(params1, data1[i]) > hmm2.marginal_log_prob(params2, data1[i]):
            correct1 += 1
    correct2 = 0
    params1, _ = hmm1.fit_em(base_params1, props1, np.array(data1[:length]), num_iters=100)
    for i in range(len(data2)):
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        params2, _ = hmm2.fit_em(base_params2, props2, np.array(temp[:length]), num_iters=100)
        if hmm1.marginal_log_prob(params1, data2[i]) < hmm2.marginal_log_prob(params2, data2[i]):
            correct2 += 1
    print(correct1,correct2)
    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])

def svmcv(data1, data2):

    zscored = zscore(np.concatenate((data1, data2)),axis=0)
    data1 = zscored[:len(data1)].tolist()
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

    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])


def permtest(data1, data2, reps=50):
    length = math.floor(np.average([len(data1), len(data2)]))
    realacc = svmcv(data1,data2)
    permaccs = []

    for i in range(reps):
        alldatas = data1 + data2
        random.shuffle(alldatas)
        data1 = alldatas[:length]
        data2 = alldatas[length:2*length]
        acc = svmcv(data1,data2)
        permaccs.append(acc)

    plt.hist(permaccs)
    plt.axvline(x=realacc)
    plt.show()

def getstats(model, params, datas):
    '''
    takes a model and a dataset, returns avg occupancy time in each state across run, avg consecutive dwell time in each
    state and avg # of transitions in a run

    Args:
        model: hidden markov model trained to the data
        datas: data from which to extract statistics

    Returns:

    '''
    num_states = np.shape(params.emissions.means)[0]
    changes = []
    occs = []
    dwells = [[]]
    for i in range(num_states - 1):
        dwells.append([])
    for data in datas:
        state = model.most_likely_states(params, data)
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
    avgoccs = np.mean(occs, axis=0)
    avgdwells = []
    for item in dwells:
        avgdwells.append(np.mean(item))
    return avgoccs, avgdwells, avgchanges

def plot_hmm_ll(data, maxstates, folds):
    test_ll = np.zeros(maxstates)
    obsdim = np.shape(data[0])[1]
    hiddenstates = np.arange(2, maxstates + 1)
    kf = KFold(n_splits=folds)
    for train, test in kf.split(data):
        for num_states in hiddenstates:
            print(num_states)
            hmm = DiagonalGaussianHMM(num_states, obsdim)
            params, props = hmm.initialize()
            params, _ = hmm.fit_em(params, props, np.array(data)[train], num_iters=100)
            for index in test:
                test_ll[num_states - 1] += hmm.marginal_log_prob(params, np.array(data)[index])
    plt.plot(hiddenstates, test_ll)
    plt.show()

def plot_trans_matrix(mat1,mat2):
    np.fill_diagonal(mat1,0)
    np.fill_diagonal(mat2,0)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(mat1)
    im = axs[1].imshow(mat2)
    axs[2].axis("off")
    plt.colorbar(im, ax=axs[2])
    plt.show()

def plot_emission_networks(mus1, mus2):
    num_states = np.shape(mus1)[0]
    obsdim = np.shape(mus1)[1]
    fig, axs = plt.subplots(2, num_states, subplot_kw=dict(projection="polar"))
    theta = np.arange(obsdim) / obsdim * 2 * math.pi
    i = 0
    while i < num_states:
        axs[0, i].set_ylim([-1.5, 1.5])
        axs[0, i].plot(theta, mus1[i, :])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_yticklabels([])
        for t, r in zip(theta, mus1[0][i, :]):
            axs[0, i].annotate(str(round(t * 17 / (2 * math.pi) + 1)), xy=[t, r])
        axs[1, i].set_ylim([-1.5, 1.5])
        axs[1, i].plot(theta, mus2[i, :])
        axs[1, i].set_xticklabels([])
        axs[1, i].set_yticklabels([])
        for t, r in zip(theta, mus2[0][i, :]):
            axs[1, i].annotate(str(round(t * 17 / (2 * math.pi) + 1)), xy=[t, r])
        i += 1
    networkstring = (
        "1 - VisCent\n "
        "2 - VisPeri\n"
        "3 - SomMotA\n"
        "4 - SomMotB\n"
        "5 - DorsAttnA\n"
        "6 - DorsAttnB\n"
        "7 - SalVentAttnA\n"
        "8 - SalVentAttnB\n"
        "9 - LimbicB\n"
        "10 - LimbicA\n"
        "11 - ContA\n"
        "12 - ContB\n"
        "13 - ContC\n"
        "14 - DefaultA\n"
        "15 - DefaultB\n"
        "16 - DefaultC\n"
        "17 - TempPar"
    )
    plt.text(-100, 0, networkstring, fontsize=10)
    plt.show()


#note—only seems to be living in very limited # of states!
#next step—compare fitting for exact same data with jax & non-jax
tues, thurs = import_tuesthurs(17)

for i in range(10):
    tues_trans = get_transmats(tues, 2)
    thurs_trans = get_transmats(thurs, 2)
    print(svmcv(tues_trans, thurs_trans))

quit()
tues_cov = []
for data in tues:
    tues_cov.append(np.cov(data.T).reshape(-1))
thurs_cov = []
for data in thurs:
    thurs_cov.append(np.cov(data.T).reshape(-1))

print(svmcv(tues_cov, thurs_cov))


quit()

accs = []
states = range(3,15)
for s in states:
    tues_trans = get_transmats(tues, s)
    thurs_trans = get_transmats(thurs, s)
    acc = svmcv(tues_trans, thurs_trans)
    accs.append(acc)
    print(acc)

plt.plot(states, accs)
plt.show()