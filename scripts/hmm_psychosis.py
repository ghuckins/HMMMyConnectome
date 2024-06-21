import numpy as np
import os
import pandas as pd
from hmm import loocv_batch, svmcv
from scipy.stats import zscore
import sys

root = "/Users/gracehuckins/PycharmProjects/HMMMyConnectome"
data_root = "/Users/gracehuckins/Documents/Research Data"

def import_all(num_networks):
    data_path = os.path.join(data_root, "psychosis")
    path = os.path.join(root, "data", f"psychosis{num_networks}")
    for filename in os.listdir(data_path):
        if filename.endswith("timeseries.tsv"):
            if (not os.path.exists(os.path.join(path, filename[:35]+"_999.txt"))
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
    for filename in os.listdir(path):
        file_key = filename[4:12]
        if filename[-5] == "9":
            if file_key in psych_data.keys():
                psych_data[file_key].append(np.loadtxt(os.path.join(path, filename)))
            else:
                psych_data[file_key] = [np.loadtxt(os.path.join(path, filename))]
        elif filename[-5] == "8":
            if file_key in hc_data.keys():
                hc_data[file_key].append(np.loadtxt(os.path.join(path, filename)))
            else:
                hc_data[file_key] = [np.loadtxt(os.path.join(path, filename))]
    return psych_data, hc_data

def import_raw(filename, num_networks):
    path = os.path.join(data_root, "psychosis")
    outliers = pd.read_table(os.path.join(path, "outliers", filename[:35] + "_outliers.tsv")).values
    if outliers[0] == 1 or outliers[-1] == 1:
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


def main():
    import_raw("sub-S0023OXB_task-rest_acq-LR_run-1_space-MNI152NLin2009cAsym_seg-4S1056Parcels_stat-mean_timeseries.tsv", 7)
    quit()
    data = pd.read_table("/Users/gracehuckins/Downloads/sub-S0023OXB_task-rest_acq-LR_run-1_space-MNI152NLin2009cAsym_seg-4S1056Parcels_stat-mean_timeseries.tsv")
    act = get_network_activity(data, 17)
    act = zscore(act)
    quit()



    quit()



    baseline = []
    reps = 10
    for rep in range(reps):
        group_1, group_2 = load_unique()
        g1_fc = []
        g2_fc = []
        for item in group_1:
            g1_fc.append(np.corrcoef(item.T)[np.triu_indices(17)])
        for item in group_2:
            g2_fc.append(np.corrcoef(item.T)[np.triu_indices(17)])
        print("data loaded")
        baseline.append(svmcv(g1_fc, g2_fc))
        print(baseline[-1])
    np.savetxt(os.path.join(root, "results", "fits", "psychosis", "baseline"), baseline)
    quit()

    path = os.path.join(root, "results", "fits")
    reps = 1
    state = int(sys.argv[1])
    accuracy = []
    for rep in range(reps):
        group_1, group_2 = load_unique()
        accuracy.append(loocv_batch(group_1, group_2, state, trans=True, ar=True, key_string="psych"))
    with open(os.path.join(path, f"psychosis{state}"), "ab") as file:
        np.savetxt(file, accuracy)

if __name__ == "__main__":
    main()