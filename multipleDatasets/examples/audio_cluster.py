import numpy as np
import scipy.io
import scipy.signal
import sys
from tqdm import tqdm
from dssplib import stft
# from paderbox.transform.module_stft import stft, istft
from matplotlib.pyplot import specgram

sys.path.insert(0, "/home/praneeth/projects/sst/git/Correlation-Analysis-in-High-Dimensional-Data/")
from multipleDatasets.correlation_analysis import MultidimensionalCorrelationAnalysis
from multipleDatasets.utils.helper import arr_sort


def most_frequent(List):
    return max(set(List), key=List.count)


def bool_to_int(x):
    y = 0
    for i, j in enumerate(x):
        y += j << i
    return y


def nd_bool2int(ndarr):
    if ndarr.shape[0] == 0:
        return 0
    else:
        nums = []
        for a in ndarr:
            nums.append(bool_to_int(a.astype(np.int32)))
        return nums


## Fetch matlab computed stft
# data_spectrum = scipy.io.loadmat('dataset/Y_zm.mat')
# Y_f_zm = data_spectrum['Y_f_zm']
# M = Y_f_zm.shape[1]


# fetch audio data from file
mat = scipy.io.loadmat('dataset/micsignals_scen_1_two_sources.mat')
mics = mat['mics']
fs_target = mat['fs_target'].squeeze()
print(mics.shape)
P = mics.shape[2]
n = mics.shape[1]
M = mics.shape[0]
Y = mics.reshape(M, P * n).T

sensors = np.array([8, 11, 5, 6, 15, 3, 9, 2, 19, 1])
# Detector parameters


B = 1000
Pfa_eval = 0.01
Pfa_evec = 0.01
n_sets = 10
thresh = 1 / np.sqrt(n_sets)
signum = 3

# zero mean the data

m_y = np.mean(Y, axis=1)
Y = Y - m_y.reshape(P * n, 1)


n1 , n2 = 300, 350  # freq range to compute on later
Y_f = []
for sig in Y:
    len = sig.shape[0]
    f, t, S_y = scipy.signal.stft(sig, nperseg=1024,window='hamming', fs=fs_target, return_onesided=True, noverlap=0)
    S_y = np.real(S_y[n1:n2,:])
    m_f = np.mean(S_y, axis=1)
    # zero mean the stft data
    S_y = S_y - m_f.reshape(m_f.shape[0], 1)
    S_y = S_y.T

    Y_f.append(S_y)

Y_f_zm = np.array(Y_f)
no_bins = Y_f_zm.shape[2]

estimator = MultidimensionalCorrelationAnalysis(n_sets, signum, M,
                                                simulation_data_type='real',  # synthetic / real
                                                Pfa_eval=Pfa_eval,
                                                Pfa_evec=Pfa_evec,
                                                threshold=thresh,  # 1/np.sqrt(10),
                                                bootstrap_count=B)

cluster_bin = np.zeros((10, 10, no_bins))
d_vec = np.zeros(no_bins)
count = 0
with tqdm(total=no_bins) as pbar:
    for b in range(no_bins):  # (no_bins):
        Y_cell = []
        for p in range(0, 30, 3):
            Y_cell.append(Y_f_zm[p:p + 3, :, b])
        struc_bin, z_bin, d_hat_i = estimator.run(Y_cell, disp_struc=False)
        nums = nd_bool2int(z_bin)
        print("d_hat= ", d_hat_i)
        print(z_bin, nums)
        d_vec[b] = d_hat_i
        cluster_bin[:d_hat_i, :, b] = z_bin
        count += 1

        pbar.update(1)

d_hat = int(np.round(np.mean(d_vec)))
print("dhat = ", d_hat)
vec_count = np.zeros(1025)
# computer majority vote:
for b in range(no_bins):
    for i in range(d_hat):
        arr = cluster_bin[i, :, b]
        idx = bool_to_int(arr.astype(np.int32))
        vec_count[idx] += 1

# pick the top d_hat after arg sorting in descending:
vec_count = vec_count
# ranked = np.argsort(vec_count)
sorted_arr, idx = arr_sort(vec_count, 'descending')
print("the hash table",vec_count)
largest_ids = idx[0:d_hat + 1]
# largest_ids = ranked[::-1][:d_hat]
print("largest ids=", largest_ids)
for ele in largest_ids:
    res = [int(x) for x in str(np.binary_repr(ele, width=10))]
    idx = np.nonzero(res)
    clusters = sensors[idx]
    print("bin form", np.binary_repr(ele, width=10), clusters)

print("experiment complete")
