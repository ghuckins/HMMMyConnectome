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

from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

root = "/Users/gracehuckins/Documents/HMMMyConnectome"
def import_all(num_networks):
    '''

    Args:
        num_networks: 7 or 17, to import 7- or 17-network parcellation

    Returns:
        data: list of numpy arrays, each array is one recording

    '''
    path = os.path.join(root,"results", f"data{num_networks}")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    data = []
    for file_name in os.listdir(path):
        data.append(np.loadtxt(os.path.join(path,file_name)))
    return data

def import_tuesthurs(num_networks,split=False):
    '''

    Args:
        num_networks: 7 or 17, to import 7- or 17-network parcellation
        split: whether to import split data

    Returns:
        tues_data: a list of numpy arrays, each of which is a recording made on tuesday
        thurs_data: a list of numpy arrays, each of which is a recording made on thursday
    '''
    if split:
        path = os.path.join(root, "results", f"data{num_networks}_split")
    else:
        path = os.path.join(root,"results",f"data{num_networks}")
    tues_data = []
    thurs_data = []
    for filename in os.listdir(path):
        if filename[6] == "t":
            tues_data.append(np.loadtxt(os.path.join(path,filename)))
        if filename[6] == "r":
            thurs_data.append(np.loadtxt(os.path.join(path,filename)))
    return tues_data, thurs_data

def import_raw(num_networks):
    print("importing")
    path = os.path.join(root,"data/MyConnectome")
    metadata = pd.read_table(os.path.join(path, "trackingdata_goodscans.txt"))
    metadata = metadata.set_index("subcode")
    savepath = os.path.join(root, "results", f"data{num_networks}")
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for filename in os.listdir(path):
        if filename.startswith("sub"):
            day = metadata.loc[filename.replace(".txt","")]["day_of_week"]
            raw_data = np.loadtxt(os.path.join(path,filename))
            raw_data = gsr(raw_data)
            activities = zscore(getNetworkActivity(raw_data, num_networks), axis=0)

            if day == "2":
                np.savetxt(os.path.join(savepath, filename.replace(".txt","t.txt")), activities)
            elif day == "4":
                np.savetxt(os.path.join(savepath, filename.replace(".txt","r.txt")), activities)
            else:
                np.savetxt(os.path.join(savepath, filename), activities)

def getNetworkActivity(data, num_networks, hcp = False):
    if hcp:
        parcellation = f"Yeo_networks{num_networks}"
        path = os.path.join(root,"data/HCP/parcel_data_hcp.txt")
    else:
        parcellation = f"{num_networks}networks"
        path = os.path.join(root,"data/MyConnectome/parcel_data.txt")
    parcels = pd.read_table(path)[parcellation]
    roughnetworks = pd.unique(parcels)
    if hcp:
        roughnetworks = np.delete(roughnetworks, np.where(roughnetworks=='No network found'))
    activities = []
    #assert that length of parcels and #features in data are the same
    for network in roughnetworks:
        if network.lower().startswith(parcellation) or hcp:
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

def split_data(num_networks):
    path = os.path.join(root,"results",f"data{num_networks}")
    dest_path = os.path.join(root,"results",f"data{num_networks}_split")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    os.mkdir(dest_path)
    for file_name in os.listdir(path):
        data = np.loadtxt(os.path.join(path,file_name))
        mask = np.loadtxt(os.path.join(root,"data","tmasks",file_name[:6]+".txt"))
        curr = mask[0]
        counter = 0
        start = 0
        for i in range(len(mask)):
            if mask[i] - curr == 1:
                start = i
                curr = mask[i]
            if mask[i] - curr == -1:
                np.savetxt(os.path.join(dest_path,file_name[:-4]+f"_{counter}.txt"), data[start:i,:])
                curr = mask[i]
                counter += 1
def get_transmats(data, latdim, ar=True, together = False):

    emissions, probs = get_saved_params(data, latdim, ar)
    hmm, params, props = init_transonly(emissions, probs, ar)

    if together:
        params = fit_all_models(hmm, params, props, jnp.array(data), ar)
        trans = params.transitions.transition_matrix

    else:
        trans = []
        for item in data:
            hmm, params, props = fit_all_models(hmm, params, props, item, ar)
            trans.append(params.transitions.transition_matrix.reshape(-1))

    return trans

def get_params_jax(data, latdim, ar, key_string="", lags=1):

    obsdim = np.shape(data[0])[1]

    if ar:
        path = os.path.join(root, "results", f"{key_string}arhmm{obsdim}")
        hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        inputs = jnp.stack([hmm.compute_inputs(datum) for datum in data])
    else:
        path = os.path.join(root,"results",f"{key_string}hmm{obsdim}")
        hmm = DiagonalGaussianHMM(latdim, obsdim)

    if not os.path.exists(path):
        os.mkdir(path)

    params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data))

    if ar:
        params, ll = hmm.fit_em(params, props, np.array(data), inputs=np.array(inputs), num_iters=1000)
    else:
        params, ll = hmm.fit_em(params, props, np.array(data), num_iters=1000)

    np.save(os.path.join(path,f"probs{latdim}"), params.initial.probs)

    with open(os.path.join(path,f"emissions{latdim}"), 'wb') as file:
        pickle.dump(params.emissions, file)

    plt.plot(range(len(ll)),ll)
    plt.show()

def get_saved_params(data, latdim, ar=True, key_string = ""):
    obsdim = np.shape(data[0])[1]

    if ar:
        path = os.path.join(root, "results", f"{key_string}arhmm{obsdim}")
    else:
        path = os.path.join(root, "results", f"{key_string}hmm{obsdim}")

    if not os.path.exists(os.path.join(path, f"emissions{latdim}")):
        print("getting parameters")
        get_params_jax(data, latdim, ar, key_string=key_string)

    with open(os.path.join(path, f"emissions{latdim}"), 'rb') as file:
        emissions = pickle.load(file)
    probs = np.load(os.path.join(path, f"probs{latdim}.npy"))

    return emissions, probs
def fit_all_models(hmm, params, props, data, ar=False, num_iters=100):
    if ar:
        inputs = jnp.stack([hmm.compute_inputs(datum) for datum in data])
        params, _ = hmm.fit_em(params, props, data, inputs=inputs, num_iters=num_iters, verbose=False)

    else:
        params, _ = hmm.fit_em(params, props, data, num_iters=num_iters, verbose=False)

    return params

def init_transonly(emissions, probs, ar, num_lags=1):

    if ar:
        latdim = emissions.weights.shape[0]
        obsdim = emissions.weights.shape[1]
        hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=num_lags)
        params, props = hmm.initialize(key=get_key(), method="prior", initial_probs=probs,
                                       emission_weights=emissions.weights, emission_biases=emissions.biases,
                                       emission_covariances=emissions.covs)
        props.emissions.weights.trainable = False
        props.emissions.biases.trainable = False
        props.emissions.covs.trainable = False

    else:
        latdim = emissions.means.shape[0]
        obsdim = emissions.means.shape[1]
        hmm = DiagonalGaussianHMM(latdim, obsdim)
        params, props = hmm.initialize(key=get_key(), method="prior", initial_probs=probs,
                                       emission_means=emissions.means, emission_scale_diags=emissions.scale_diags)
        props.emissions.means.trainable = False
        props.emissions.scale_diags.trainable = False

    props.initial.probs.trainable = False

    return hmm, params, props

def logprob_all_models(hmm,params,data,ar):
    if ar:
        inputs = hmm.compute_inputs(data)
        return hmm.marginal_log_prob(params, data, inputs=inputs)
    return hmm.marginal_log_prob(params, data)

def loocv(data1, data2, latdim, trans=False, ar=False, lags=1, key_string=""):

    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    assert np.shape(data2[0])[1] == obsdim, "Datasets must have the same number of features"
    length = min(len(data1), len(data2)) - 1

    if trans:
        emissions, probs = get_saved_params(np.concatenate((data1, data2), axis=0), latdim, ar, key_string=key_string)
        hmm, base_params1, props = init_transonly(emissions, probs, ar)
        base_params2 = base_params1

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

        base_params1, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data1[:length]))
        base_params2, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data2[:length]))

    correct1 = 0
    params2 = fit_all_models(hmm, base_params2, props, np.array(data2[:length]), ar)
    for i in range(len(data1)):
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        params1 = fit_all_models(hmm, base_params1, props, np.array(temp[:length]), ar)
        if logprob_all_models(hmm, params1, data1[i], ar) > logprob_all_models(hmm, params2, data1[i], ar):
            correct1 += 1
    correct2 = 0
    params1 = fit_all_models(hmm, base_params1, props, np.array(data1[:length]), ar)
    for i in range(len(data2)):
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        params2 = fit_all_models(hmm, base_params2, props, np.array(temp[:length]), ar)
        if logprob_all_models(hmm, params1, data2[i], ar) < logprob_all_models(hmm, params2, data2[i], ar):
            correct2 += 1
    print(np.average([correct1 / (correct1 + len(data2) - correct2), correct2 / (correct2 + len(data1) - correct1)]))
    return np.average([correct1 / (correct1 + len(data2) - correct2), correct2 / (correct2 + len(data1) - correct1)])
def loocv_batch(data1, data2, latdim, trans=False, ar=False, lags=1, key_string=""):
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    length = min(len(data1), len(data2)) - 1

    data1 = jnp.array(data1)
    data2 = jnp.array(data2)
    data1_train = jnp.stack([jnp.concatenate([data1[:i], data1[i + 1:]]) for i in range(len(data1))])
    data2_train = jnp.stack([jnp.concatenate([data2[:i], data2[i + 1:]]) for i in range(len(data2))])

    if trans:
        emissions, probs = get_saved_params(np.concatenate((data1, data2), axis=0), latdim, ar, key_string=key_string)
        hmm, params, props = init_transonly(emissions, probs, ar)

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, obsdim)

        params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=data1[:length, :, :])

    params1 = fit_all_models(hmm, params, props, data1[:length, :, :], ar)
    if not trans:
        params, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=data2[:length, :, :])
    params2 = fit_all_models(hmm, params, props, data2[:length,:,:], ar)
    def _fit_fold(train, test, comp_params):
        para = params
        pr = props
        if not trans:
            para, _ = kmeans_init(hmm, train[:length,:,:], get_key(), ar)
        fit_params = fit_all_models(hmm, para, pr, train[:length,:,:], ar)
        return (logprob_all_models(hmm, fit_params, test, ar) > logprob_all_models(hmm, comp_params, test, ar)).astype(int)

    correct1 = jnp.sum(vmap(_fit_fold, in_axes=[0, 0, None])(data1_train, data1, params2))
    correct2 = jnp.sum(vmap(_fit_fold, in_axes=[0, 0, None])(data2_train, data2, params1))
    print(np.average([correct1 / (correct1 + len(data2) - correct2), correct2 / (correct2 + len(data1) - correct1)]))
    return np.average([correct1 / (correct1 + len(data2) - correct2), correct2 / (correct2 + len(data1) - correct1)])
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

    print(np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)]))
    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])

def permtest(data1, data2, class_func, latdim=6, reps=50):
    length = math.floor(np.average([len(data1), len(data2)]))
    if class_func == svmcv:
        realacc = svmcv(data1,data2)
    else:
        realacc = class_func(data1, data2, latdim)
    permaccs = []

    for i in range(reps):
        alldatas = data1 + data2
        random.shuffle(alldatas)
        data1 = alldatas[:length]
        data2 = alldatas[length:2*length]
        if class_func == svmcv:
            acc = svmcv(data1, data2)
        else:
            acc = class_func(data1, data2, latdim)

        permaccs.append(acc)

    plt.hist(permaccs,bins=50)
    plt.axvline(x=realacc)
    plt.show()
    return np.average(permaccs)

def getstats(model, params, datas, num_states, ar):
    '''
    takes a model and a dataset, returns avg occupancy time in each state across run, avg consecutive dwell time in each
    state and avg # of transitions in a run

    Args:
        model: hidden markov model trained to the data
        datas: data from which to extract statistics

    Returns:

    '''
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
            state = model.most_likely_states(params,data,inputs = input)
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
        return occs#avgoccs, avgdwells, avgchanges, stdoccs, stddwells, stdchanges

def get_key():
    return jr.PRNGKey(random.randint(0,10000))
def plot_hmm_ll(data, maxstates, folds):
    test_ll = np.zeros(maxstates-1)
    print(len(test_ll))
    obsdim = np.shape(data[0])[1]
    hiddenstates = np.arange(2, maxstates + 1)
    print(len(hiddenstates))
    kf = KFold(n_splits=folds)
    for train, test in kf.split(data):
        for num_states in hiddenstates:
            print(num_states)
            #hmm = DiagonalGaussianHMM(num_states, obsdim)
            hmm = LinearAutoregressiveHMM(num_states, obsdim, num_lags=1)
            params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(data)[train])
            inputs = []
            for index in train:
                inputs.append(hmm.compute_inputs(np.array(data)[index]))
            params, _ = hmm.fit_em(params, props, np.array(data)[train], inputs=np.array(inputs), num_iters=100, verbose=False)
            for index in test:
                inputs = hmm.compute_inputs(np.array(data)[index])
                test_ll[num_states - 2] += hmm.marginal_log_prob(params, np.array(data)[index], inputs=inputs)
    plt.plot(hiddenstates, test_ll)
    plt.show()

def plot_avg_lls():
    path = os.path.join(root,"results","fits")
    full7 = [np.loadtxt(os.path.join(path,"7diagfull.txt"))[:11]]
    trans7 = [np.loadtxt(os.path.join(path,"7diagtrans.txt"))[:11]]
    full17 = [np.loadtxt(os.path.join(path,"17diagfull.txt"))[:11]]
    trans17 = [np.loadtxt(os.path.join(path,"17diagtrans.txt"))[:11]]
    for file_name in os.listdir(path):
        if file_name.startswith("7full"):
            full7.append(np.loadtxt(os.path.join(path,file_name)))
        elif file_name.startswith("7trans"):
            trans7.append(np.loadtxt(os.path.join(path,file_name)))
        elif file_name.startswith("17full"):
            full17.append(np.loadtxt(os.path.join(path,file_name)))
        elif file_name.startswith("17trans"):
            trans17.append(np.loadtxt(os.path.join(path,file_name)))
    base7 = np.loadtxt(os.path.join(path,"7baseline.txt"))
    base17 = np.loadtxt(os.path.join(path,"17baseline.txt"))
    #plt.errorbar(np.arange(2,13),np.mean(full7,axis=0),yerr=np.std(full7,axis=0),label="full")
    plt.errorbar(np.arange(2,13),np.mean(full17,axis=0),yerr=np.std(full7,axis=0),label="full")
    #plt.errorbar(np.arange(2, 13), np.mean(trans7, axis=0),yerr=np.std(full7,axis=0), label="trans only")
    plt.errorbar(np.arange(2, 13), np.mean(trans17, axis=0),yerr=np.std(full7,axis=0), label="trans only")
    #plt.errorbar(np.arange(2, 13), np.ones(11)*np.mean(base7), yerr=np.ones(11)*np.std(base7), label="baseline")
    plt.errorbar(np.arange(2, 13), np.ones(11) * np.mean(base17), yerr=np.ones(11) * np.std(base17),label="baseline")
    plt.legend()
    plt.xlabel("hidden states")
    plt.ylabel("classification accuracy")
    plt.title("17 networks")
    plt.show()

def plot_trans_matrix(mat1,mat2):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    np.fill_diagonal(mat1,0)
    np.fill_diagonal(mat2,0)
    sns.set(font_scale=0.75)
    fig, ax = plt.subplots(ncols=3)
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([.88, .345, .03, .3])
    sns.heatmap(mat1, cbar_ax=cbar_ax, ax=ax[0], vmin=0, vmax=0.1, square=True, cmap="viridis")
    sns.heatmap(mat2, cbar_ax=cbar_ax, ax=ax[1], vmin=0, vmax=0.1, square=True, cmap="viridis")
    sns.heatmap(np.abs(mat2 - mat1), cbar_ax = cbar_ax, vmin=0, vmax=0.1,ax=ax[2], square=True, cmap="viridis")
    ax[0].set_title("Uncaffeinated \nRuns", fontsize=10)
    ax[1].set_title("Caffeinated \nRuns", fontsize=10)
    ax[2].set_title("Absolute \nDifference",fontsize=10)
    plt.savefig("./results/figs/OHBM trans matrices TRANSPARENT", facecolor=(1, 1, 1, 0))
    plt.show()

def plot_emission_networks(mus1):#, mus2):
    num_states = np.shape(mus1)[0]
    obsdim = np.shape(mus1)[1]
    fig, axs = plt.subplots(1, num_states, subplot_kw=dict(projection="polar"))
    theta = np.arange(obsdim) / obsdim * 2 * math.pi
    i = 0
    while i < num_states:
        axs[i].set_ylim([-1.5, 1.5])
        axs[i].plot(theta, mus1[i, :])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        for t, r in zip(theta, mus1[i, :]):
            axs[i].annotate(str(round(t * 17 / (2 * math.pi) + 1)), xy=[t, r])
        #axs[1, i].set_ylim([-1.5, 1.5])
        #axs[1, i].plot(theta, mus2[i, :])
        #axs[1, i].set_xticklabels([])
        #axs[1, i].set_yticklabels([])
        #for t, r in zip(theta, mus2[0][i, :]):
        #    axs[1, i].annotate(str(round(t * 17 / (2 * math.pi) + 1)), xy=[t, r])
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
def main():


    tues, thurs = import_tuesthurs(17)
    print(loocv(tues, thurs, 6, trans=False, ar=True))

    quit()


    with open(os.path.join(root, "results", "fits", "hmm7_split", "allgaussiandata"), "rb") as file:
        data = pickle.load(file)

    sns.set_theme()
    colors = [[51 / 255, 34 / 255, 136 / 255], [136 / 255, 204 / 255, 238 / 255], [17 / 255, 119 / 255, 51 / 255],
              [153 / 255, 153 / 255, 51 / 255], [204 / 255, 102 / 255, 119 / 255], [136 / 255, 34 / 255, 85 / 255]]
    sns.set_palette(sns.color_palette(colors))
    fig = sns.relplot(data=data, x="Hidden States", y="Classification Accuracy", hue="Head Motion", kind="line",
                      errorbar="ci")
    sns.move_legend(fig, "upper right", bbox_to_anchor=(0.717, 0.73))
    fig.legend.set_title(None)
    fig.legend.set(frame_on=True)

    fig.fig.subplots_adjust(top=.9)
    plt.ylim([0.4, 1])
    plt.xticks([2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.title("Classification accuracy on MyConnectome data,\ncensored vs. uncensored head motion", weight='bold',
              fontsize=14)

    fig.tight_layout()
    plt.savefig("./results/figs/head_motion_test", facecolor=(1, 1, 1, 0))
    plt.show()
    quit()


    acc = []
    model = []
    states = []
    path = os.path.join(root, "results", "fits", "psychosis")
    for filename in os.listdir(path):
        data = np.loadtxt(os.path.join(path, filename))
        if filename[:1] == "b":
            for r in range(2,11):
                acc.extend(data)
                model.extend(["Static Baseline"]*len(data))
                states.extend([r]*len(data))
        elif int(filename[9:]) < 11:
            model.extend(["Dynamic Classifier"]*len(data))
            states.extend([int(filename[9:])]*len(data))
            acc.extend(data)

    data = pd.DataFrame({"Classification Accuracy": acc, "Model": model, "Hidden States": states})
    with open(os.path.join(path, "alldata"), "wb") as file:
        pickle.dump(data, file)



    hm_fc = {}
    hm_fc["t"] = []
    hm_fc["r"] = []
    nohm_fc = {}
    nohm_fc["t"] = []
    nohm_fc["r"] = []

    path1 = os.path.join(root,'results','data7_split')
    path2 = os.path.join(root,'results','data7')
    thresh = 100

    for filename in os.listdir(path2):
        if filename[6] != "t" and filename[6] != "r":
            continue

        data = np.loadtxt(os.path.join(path2,filename))
        num = 0
        hm_pos = 0
        while os.path.exists(os.path.join(path1, filename[:7] + f"_{num}.txt")):
            data1 = np.loadtxt(os.path.join(path1, filename[:7] + f"_{num}.txt"))
            counter = 1
            while counter <= len(data1) / thresh:
                hm_pos += 1
                nohmdata = data1[(counter - 1) * thresh:(counter) * thresh]
                hmdata = data[(hm_pos - 1) * thresh:(hm_pos) * thresh]
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
    np.savetxt(os.path.join(root, "results", "fits", "hmm7_split", "bl_motion"), bl_motion)
    np.savetxt(os.path.join(root, "results", "fits", "hmm7_split", "bl_nomotion"), bl_nomotion)
    quit()





    tues, thurs = import_tuesthurs(17)
    all_data = tues.copy()
    all_data.extend(thurs)

    emissions, probs = get_saved_params(tues, 4, True)
    hmm, params, props = init_transonly(emissions, probs, True)
    params = fit_all_models(hmm, params, props, np.array(all_data), True)

    sns.set_theme()
    colors = [[51 / 255, 34 / 255, 136 / 255], [136 / 255, 204 / 255, 238 / 255], [17 / 255, 119 / 255, 51 / 255],
              [153 / 255, 153 / 255, 51 / 255], [204 / 255, 102 / 255, 119 / 255], [136 / 255, 34 / 255, 85 / 255]]
    sns.set_palette(sns.color_palette(colors))


    tuesoccs = np.array(getstats(hmm,params,tues,4, True))/518
    thursoccs = getstats(hmm,params,thurs,4, True)
    tuesoccs = tuesoccs.reshape(-1,1)
    tuesstates = np.array([[0,1,2,3]*30]).T
    thursoccs = np.array(getstats(hmm,params,thurs,4, True))/518
    thursoccs = thursoccs.reshape(-1,1)
    alloccs = np.concatenate((tuesoccs,thursoccs),axis=0)
    thursstates = np.array([[0,1,2,3]*23]).T
    allstates = np.concatenate((tuesstates,thursstates),axis=0)
    labels = ["Uncaffeinated"]*120
    labels.extend(["Caffeinated"]*92)
    occsdict = {}
    occsdict["Occupancy"] = alloccs.flatten()
    occsdict["State"] = allstates.flatten()
    occsdict["Label"] = labels
    print(len(occsdict["Occupancy"]))
    print(len(occsdict["State"]))
    print(len(occsdict["Label"]))
    occsdf = pd.DataFrame(data=occsdict)
    ax = sns.barplot(occsdf,x="State",y="Occupancy",hue="Label",estimator="mean",errorbar="ci")
    #fig.legend.set_title(None)
    ax.legend(title=None)
    plt.savefig("./results/figs/OHBM occupancies TRANSPARENT", facecolor=(1, 1, 1, 0))
    plt.show()

    #print(t)
    quit()




    path = os.path.join(root, "results", "fits")
    with open(os.path.join(path, "dataframeall"), "rb") as file:
        data = pickle.load(file)

    sns.set_theme()
    colors = [[51 / 255, 34 / 255, 136 / 255], [136 / 255, 204 / 255, 238 / 255], [17 / 255, 119 / 255, 51 / 255],
              [153 / 255, 153 / 255, 51 / 255], [204 / 255, 102 / 255, 119 / 255], [136 / 255, 34 / 255, 85 / 255]]
    sns.set_palette(sns.color_palette(colors))
    fig = sns.relplot(data=data, x="Hidden States", y="Classification Accuracy", hue="Model", col="Networks", kind="line",
                      errorbar="ci").set_titles('7 Networks', weight='bold', size=14)
    sns.move_legend(fig,"upper right",bbox_to_anchor=(0.817, 0.93))
    fig.legend.set_title(None)
    fig.legend.set(frame_on=True)

    fig.fig.subplots_adjust(top=.9)
    plt.ylim([0.6, 1])
    plt.title("17 Networks", weight='bold', fontsize=14)

    fig.tight_layout()
    plt.savefig("./results/figs/OHBM HCP All TRANSPARENT", facecolor=(1, 1, 1, 0))
    plt.show()
    quit()

    acc = []
    state = []
    mod = []
    network = []

    networks = [7,17]

    for net in networks:
        for file_name in os.listdir(os.path.join(path, f'hmm{net}')):
            acc.extend(np.loadtxt(os.path.join(path, f'hmm{net}', file_name))[:100])
            state.extend([int(file_name[3:])] * 100)
            mod.extend(["Full, HMM"] * 100)
            network.extend([f"{net} Networks"]*100)
        for file_name in os.listdir(os.path.join(path, f'trans{net}')):
            acc.extend(np.loadtxt(os.path.join(path, f'trans{net}', file_name))[:100])
            state.extend([int(file_name[3:])] * 100)
            mod.extend(["Trans Only, HMM"] * 100)
            network.extend([f"{net} Networks"] * 100)
        for file_name in os.listdir(os.path.join(path, f'arhmm{net}')):
            acc.extend(np.loadtxt(os.path.join(path, f'arhmm{net}', file_name))[:100])
            state.extend([int(file_name[3:])] * 100)
            mod.extend(["Full, ARHMM"] * 100)
            network.extend([f"{net} Networks"] * 100)
        for file_name in os.listdir(os.path.join(path, f'artrans{net}')):
            acc.extend(np.loadtxt(os.path.join(path, f'artrans{net}', file_name))[:100])
            state.extend([int(file_name[3:])] * 100)
            mod.extend(["Trans Only, ARHMM"] * 100)
            network.extend([f"{net} Networks"] * 100)
        baseline = np.loadtxt(os.path.join(path, f'baseline{net}.txt'))
        for hs in np.arange(2, 13):
            acc.extend(baseline)
            state.extend([hs] * len(baseline))
            mod.extend(["Baseline"] * len(baseline))
            network.extend([f"{net} Networks"] * 100)

    datadict = {}
    datadict["Classification Accuracy"] = acc
    datadict["Hidden States"] = state
    datadict["Model"] = mod
    datadict["Networks"] = network
    data = pd.DataFrame(data=datadict)
    with open(os.path.join(path, "dataframeall"), "wb") as file:
        pickle.dump(data, file)

    quit()






    path = os.path.join(root, "results", "fits")

    path = os.path.join(root, "results", "fits")

    with open(os.path.join(path, "dataframe7"), "rb") as file:
        data = pickle.load(file)

    sns.relplot(data=data, x="Hidden States", y="Classification Accuracy", hue="Model", kind="line", errorbar="sd")
    plt.ylim([0.5, 1])
    plt.show()
    quit()

    with open(os.path.join(path, "dataframe17"), "rb") as file:
        data = pickle.load(file)

    sns.relplot(data=data, x="Hidden States", y="Classification Accuracy", hue="Model", kind="line", errorbar="sd")
    plt.ylim([0.5,1])
    plt.show()
    quit()


    acc = []
    state = []
    mod = []

    for file_name in os.listdir(os.path.join(path,'hmm17')):
        acc.extend(np.loadtxt(os.path.join(path,'hmm17',file_name))[:100])
        state.extend([int(file_name[3:])]*100)
        mod.extend(["Full, HMM"]*100)
    for file_name in os.listdir(os.path.join(path,'arhmm17')):
        acc.extend(np.loadtxt(os.path.join(path,'arhmm17',file_name))[:100])
        state.extend([int(file_name[3:])]*100)
        mod.extend(["Full, ARHMM"]*100)
    for file_name in os.listdir(os.path.join(path,'trans17')):
        acc.extend(np.loadtxt(os.path.join(path,'trans17',file_name))[:100])
        state.extend([int(file_name[3:])]*100)
        mod.extend(["Trans Only, HMM"]*100)
    for file_name in os.listdir(os.path.join(path,'artrans17')):
        acc.extend(np.loadtxt(os.path.join(path,'artrans17',file_name))[:100])
        state.extend([int(file_name[3:])]*100)
        mod.extend(["Trans Only, ARHMM"]*100)
    baseline = np.loadtxt(os.path.join(path,'baseline17.txt'))
    for hs in np.arange(2,13):
        acc.extend(baseline)
        state.extend([hs]*len(baseline))
        mod.extend(["Baseline"]*len(baseline))

    datadict = {}
    datadict["Classification Accuracy"] = acc
    datadict["Hidden States"] = state
    datadict["Model"] = mod
    data7 = pd.DataFrame(data=datadict)
    with open(os.path.join(path,"dataframe17"),"wb") as file:
        pickle.dump(data7,file)

    batch_results = []
    full_results = []

    tues, thurs = import_tuesthurs(7)

    reps = 10

    for i in range(reps):
        batch_results.append(loocv_batch(tues, thurs, 4, trans=True, ar=True))
        full_results.append(loocv(tues, thurs, 4, trans=True, ar=True))

    print(batch_results)
    print(full_results)
    plt.scatter(np.zeros(reps),batch_results,label="jax runs")
    plt.scatter(np.ones(reps),full_results,label="full runs")
    plt.legend()
    plt.show()
    quit()


    iterations = np.arange(6,9)
    tues7, thurs7 = import_tuesthurs(7)
    tues17, thurs17 = import_tuesthurs(17)
    states = np.arange(2, 13)
    for iter in iterations:
        print(f'Iteration {iter}')
        full7 = []
        full17 = []
        trans7 = []
        trans17 = []
        for state in states:
            print(f'State {state}')
            full7.append(loocv(tues7,thurs7,state))
            full17.append(loocv(tues17,thurs17,state))
            trans7.append(loocvtrans(tues7,thurs7,state))
            trans17.append(loocvtrans(tues17,thurs17,state))
        np.savetxt(f'7full{iter}.txt', full7)
        np.savetxt(f'17full{iter}.txt', full17)
        np.savetxt(f'7trans{iter}.txt', trans7)
        np.savetxt(f'17trans{iter}.txt', trans17)



    tues, thurs = import_tuesthurs(17)
    tues_fc = []
    thurs_fc = []
    for item in tues:
        tues_fc.append(np.corrcoef(item.T)[np.triu_indices(17)])
    for item in thurs:
        thurs_fc.append(np.corrcoef(item.T)[np.triu_indices(17)])

    acc = []
    for i in range(10):
        acc.append(svmcv(tues_fc,thurs_fc))
    np.savetxt("17baseline.txt",acc)
    print(acc)

if __name__ == "__main__":
    main()



