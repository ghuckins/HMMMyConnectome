import numpy as np
import os
from scipy.stats import zscore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from hmm import gsr, getNetworkActivity, get_saved_params, get_key,\
    fit_all_models, init_transonly, logprob_all_models
from k_means import kmeans_init
import random
import jax.numpy as jnp
from jax import vmap
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

from src.dynamax import DiagonalGaussianHMM
from src.dynamax import LinearAutoregressiveHMM

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
            if num_networks == 512:
                activities = zscore(raw_data, axis=0)
            else:
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
        file_key = int(file[3:9])
        if file_key in data.keys():
            data[file_key].append(np.loadtxt(os.path.join(dir,file)))
        else:
            data[file_key] = [np.loadtxt(os.path.join(dir,file))]
    return data

def alt_params(data, latdim, trans=False, ar=False):
    '''

    Args:
        data:
        latdim:
        trans:
        ar:

    Returns:
        a dict of fit parameters for each subject
    '''

    num_networks = np.shape(list(data.values())[0])[2]

    ar_str = ""
    trans_str = ""
    if ar:
        ar_str = "ar"
    if trans:
        trans_str = "trans"
    dir = os.path.join(root, "results", f"hcpparams{ar_str}{trans_str}{num_networks}")

    if not os.path.exists(dir):
        os.mkdir(dir)

    filepath = os.path.join(dir,f"params{latdim}")
    if not os.path.exists(filepath):
        print(f"fitting params {latdim}{trans}{ar}")
        params = {}

        if trans:
            emissions, probs = get_saved_params([i for j in data.values() for i in j], latdim, ar, key_string="hcp")
            hmm, base_params, props = init_transonly(emissions, probs, ar)
        else:
            if ar:
                hmm = LinearAutoregressiveHMM(latdim, num_networks)
            else:
                hmm = DiagonalGaussianHMM(latdim, num_networks)

        for key in data.keys():
            params[key] = []
            curr_data = jnp.array(data[key])
            for i in range(len(curr_data)):
                temp_data = jnp.concatenate([curr_data[:i], curr_data[i + 1:]])
                if not trans:
                    base_params, props = kmeans_init(hmm, curr_data, get_key(), ar)
                curr_params = fit_all_models(hmm, base_params, props, temp_data, ar)
                params[key].append(curr_params)

        with open(filepath, "wb") as file:
            pickle.dump(params, file)

    else:
        with open(filepath,"rb") as file:
            params = pickle.load(file)

    return params


def loohcp(data, latdim, num_subjs, trans=False, ar=False, lags=1):

    num_networks = np.shape(list(data.values())[0])[2]

    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    params = {}

    if trans:
        emissions, probs = get_saved_params([i for j in data.values() for i in j], latdim, ar, key_string="hcp")
        hmm, base_params, props = init_transonly(emissions, probs, ar)

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, num_networks, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, num_networks)

    correct = 0
    for key in keys:
        train_data = data[key]
        random.shuffle(train_data)
        train_data = train_data[:-1]
        if not trans:
            base_params, props = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(train_data))
        params[key] = fit_all_models(hmm, base_params, props, np.array(train_data), ar)
    for key in keys:
        i = 0
        while i < len(data[key]):
            train_data = data[key].copy()
            train_data.pop(i)
            if not trans:
                base_params, _ = hmm.initialize(key=get_key(), method="kmeans", emissions=np.array(train_data))
            loo_params = fit_all_models(hmm, base_params, props, np.array(train_data), ar)
            log_likelihoods = [logprob_all_models(hmm, loo_params, data[key][i], ar)]
            for key2 in keys:
                if key2 != key:
                    log_likelihoods.append(logprob_all_models(hmm,params[key2], data[key][i], ar))
            if np.argmax(log_likelihoods) == 0:
                correct += 1
            i = i + 1
    print(correct/(num_subjs*4))
    return correct/(num_subjs*4)
def loohcp_batch(data, latdim, num_subjs, trans=False, ar=False, lags=1):

    num_networks = np.shape(list(data.values())[0])[2]

    keys = list(data.keys()) #list of keys for all subjects in dataset
    random.shuffle(keys)
    keys = keys[:num_subjs] #randomly select num_subjs subjects
    params = alt_params(data, latdim, trans, ar)

    if trans:
        emissions, probs = get_saved_params([i for j in data.values() for i in j], latdim, ar, key_string="hcp")
        hmm, b_p, pr = init_transonly(emissions, probs, ar)

    else:
        if ar:
            hmm = LinearAutoregressiveHMM(latdim, num_networks, num_lags=lags)
        else:
            hmm = DiagonalGaussianHMM(latdim, num_networks)

    train = []
    test = []
    loo_keys = []
    key_indices = jnp.arange(num_subjs)
    for j in key_indices:
        curr_data = data[keys[j]]
        test.extend(curr_data)
        curr_data = jnp.array(curr_data)
        train.extend([jnp.concatenate([curr_data[:i], curr_data[i + 1:]]) for i in range(4)])
        loo_keys.extend([jnp.concatenate([key_indices[:j], key_indices[j + 1:]])]*4)

    train = jnp.stack(train)
    print(np.shape(train))
    test = jnp.stack(test)
    print(np.shape(test))
    def _fit_fold(train, test, alt_params):
        if trans:
            base_params = b_p
            props = pr
        else:
            base_params, props = kmeans_init(hmm, train, get_key(), ar)
        loo_params = fit_all_models(hmm, base_params, props, train, ar)
        ll = [logprob_all_models(hmm, loo_params, test, ar)]
        ll.extend([logprob_all_models(hmm, alt_params[i], test, ar) for i in range(num_subjs)])
        return ll#jnp.argsort(-jnp.array(ll))

    param_list = [random.choice(params[keys[i]]) for i in range(num_subjs)]
    print(param_list[3])
    ll_sort = vmap(_fit_fold, in_axes=[0, 0, None])(train, test, param_list)
    #print(ll_sort)
    quit()
    indices = jnp.repeat(jnp.arange(num_subjs), 4)+1
    correct = jnp.sum((ll_sort[:, 0] == 0).astype(int)) + jnp.dot(jnp.equal(indices,ll_sort[:,0]).astype(int), (ll_sort[:,1] == 0).astype(int))
    return correct/(num_subjs*4)

def baselineFingerprint(num_networks, num_subjs):
    data = load_hcp(num_networks)
    keys = list(data.keys())
    random.shuffle(keys)
    keys = keys[:num_subjs]
    compare_cov = {}
    for key in keys: #getting baseline average cov matrices from 3 runs for all participants
        compare_data = data[key]
        random.shuffle(compare_data)
        compare_data = compare_data[:-1]
        compare_data = np.array([np.corrcoef(item.T) for item in compare_data])
        compare_cov[key] = np.average(compare_data,axis=0)
    correct = 0
    for key in keys:
        i = 0
        score = 1
        while i < len(data[key]):
            train_data = data[key].copy()
            train_data.pop(i)
            train_cov = np.average(np.array([np.corrcoef(item.T) for item in train_data]),axis=0)
            cov = np.corrcoef(data[key][i].T)
            max_corr = np.corrcoef(train_cov.flatten(),cov.flatten())[0,1]
            j = 0
            while j < len(keys) and score == 1:
                corr = np.corrcoef(cov.flatten(),compare_cov[keys[j]].flatten())[0,1]
                if corr > max_corr and keys[j] != key:
                    score = 0
                j = j + 1
            correct += score
            i = i + 1
    print(correct/(4*num_subjs))
    return correct/(4*num_subjs)

def main():

    print("code working")
    quit()

    data = load_hcp(17)
    print(loohcp(data,5,10,trans=False,ar=True,lags=1))
    quit()

    path = os.path.join(root, "results", "fits", "hcp")
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
    plt.ylim([0, 1])
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
            fits = np.loadtxt(os.path.join(path, f'hmm{net}', file_name))[:50]
            acc.extend(fits)
            state.extend([int(file_name[3:])] * len(fits))
            mod.extend(["Full, HMM"] * len(fits))
            network.extend([f"{net} Networks"]*len(fits))
        for file_name in os.listdir(os.path.join(path, f'trans{net}')):
            fits = np.loadtxt(os.path.join(path, f'trans{net}', file_name))[:50]
            acc.extend(fits)
            state.extend([int(file_name[3:])] * len(fits))
            mod.extend(["Trans Only, HMM"] * len(fits))
            network.extend([f"{net} Networks"] * len(fits))
        for file_name in os.listdir(os.path.join(path, f'arhmm{net}')):
            fits = np.loadtxt(os.path.join(path, f'arhmm{net}', file_name))[:50]
            acc.extend(fits)
            state.extend([int(file_name[3:])] * len(fits))
            mod.extend(["Full, ARHMM"] * len(fits))
            network.extend([f"{net} Networks"] * len(fits))
        for file_name in os.listdir(os.path.join(path, f'artrans{net}')):
            fits = np.loadtxt(os.path.join(path, f'artrans{net}', file_name))[:50]
            acc.extend(fits)
            state.extend([int(file_name[3:])] * len(fits))
            mod.extend(["Trans Only, ARHMM"] * len(fits))
            network.extend([f"{net} Networks"] * len(fits))
        baseline = np.loadtxt(os.path.join(path, f'baselinef{net}.txt'))[:50]
        for hs in np.arange(2, 13):
            acc.extend(baseline)
            state.extend([hs] * len(baseline))
            mod.extend(["Baseline"] * len(baseline))
            network.extend([f"{net} Networks"] * len(baseline))
        big_baseline = np.loadtxt(os.path.join(path, f'baselinef512.txt'))[:50]
        for hs in np.arange(2, 13):
            acc.extend(big_baseline)
            state.extend([hs] * len(big_baseline))
            mod.extend(["Baseline (512 ROIs)"] * len(big_baseline))
            network.extend([f"{net} Networks"] * len(big_baseline))

    datadict = {}
    datadict["Classification Accuracy"] = acc
    datadict["Hidden States"] = state
    datadict["Model"] = mod
    datadict["Networks"] = network
    data = pd.DataFrame(data=datadict)
    with open(os.path.join(path, "dataframeall"), "wb") as file:
        pickle.dump(data, file)

    quit()




    reps = 100
    bf = []
    for rep in range(reps):
        bf.append(baselineFingerprint(17, 100))
    with open(os.path.join(path,"baselinef17.txt"),"ab") as file:
        np.savetxt(file,bf)
    quit()

    path = os.path.join(root, "results", "fits", "hcp")

    data = load_hcp(17)

    reps = 20
    min_states = 6
    states = np.arange(min_states, 13)
    for state in states:
        full = []
        full_ar = []
        trans = []
        trans_ar = []
        for rep in range(reps):
            print(state,rep)
            full.append(loohcp_batch(data, state, 100, trans=False, ar=False))
            full_ar.append(loohcp_batch(data, state, 100, trans=False, ar=True))
            trans.append(loohcp_batch(data, state, 100, trans=True, ar=False))
            trans_ar.append(loohcp_batch(data, state, 100, trans=True, ar=True))

        with open(os.path.join(path, "hmm17", f"fit{state}"), "ab") as file:
            np.savetxt(file, full)
        with open(os.path.join(path, "arhmm17", f"fit{state}"), "ab") as file:
            np.savetxt(file, full_ar)
        with open(os.path.join(path, "trans17", f"fit{state}"), "ab") as file:
            np.savetxt(file, trans)
        with open(os.path.join(path, "artrans17", f"fit{state}"), "ab") as file:
            np.savetxt(file, trans_ar)

    data = load_hcp(17)

    states = np.arange(2,13)
    for state in states:
        alt_params(data, state, trans=False, ar=False)
        alt_params(data, state, trans=False, ar=True)
        alt_params(data, state, trans=True, ar=False)
        alt_params(data, state, trans=True, ar=True)

    quit()








    reps = 10
    min_states = int(sys.argv[1])
    states = np.arange(min_states,13)
    for state in states:
        full = []
        full_ar = []
        trans = []
        trans_ar = []
        for rep in range(reps):
            full.append(loohcp_batch(data, state, 100, trans=False, ar=False))
            full_ar.append(loohcp_batch(data, state, 100, trans=False, ar=True))
            trans.append(loohcp_batch(data, state, 100, trans=True, ar=False))
            trans_ar.append(loohcp_batch(data, state, 100, trans=True, ar=True))

        with open(os.path.join(path, "hmm7", f"fit{state}"), "ab") as file:
            np.savetxt(file, full)
        with open(os.path.join(path, "arhmm7", f"fit{state}"), "ab") as file:
            np.savetxt(file, full_ar)
        with open(os.path.join(path, "trans7", f"fit{state}"), "ab") as file:
            np.savetxt(file, trans)
        with open(os.path.join(path, "artrans7", f"fit{state}"), "ab") as file:
            np.savetxt(file, trans_ar)


if __name__ == "__main__":
    main()
