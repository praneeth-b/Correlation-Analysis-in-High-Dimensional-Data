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

def most_frequent(List):
    return max(set(List), key=List.count)

def bool_to_int(x):
    y=0
    for i,j in enumerate(x):
        y += j<<i
    return y

# fetch audio data from file
mat = scipy.io.loadmat('dataset/micsignals_scen_1.mat')
mics = mat['mics']
fs_target = mat['fs_target'].squeeze()
print(mics.shape)
P = mics.shape[2]
n = mics.shape[1]
M = mics.shape[0]
Y = mics.reshape(M, P * n).T
# Detector parameters
thresh = 1 / np.sqrt(P)
B = 1000
Pfa_eval = 0.01
Pfa_evec = 0.01
n_sets = 10
signum = 3
tot_dims = 10

data_spectrum = scipy.io.loadmat('dataset/Y_zm.mat')
Y_f_zm = data_spectrum['Y_f_zm']
M = Y_f_zm.shape[1]

estimator = MultidimensionalCorrelationAnalysis(n_sets, signum, M,
                                                simulation_data_type='real',  # synthetic / real
                                                Pfa_eval=Pfa_eval,
                                                Pfa_evec=Pfa_evec,
                                                threshold=thresh,  # 1/np.sqrt(10),
                                                bootstrap_count=B)

# # zero mean the data
#
# m_y = np.mean(Y,axis=1)
# Y = Y - m_y.reshape(P*n, 1)
# STFT parameters:

# no_bins = 32  # freq bins
# frame_size = 64   # window size
# fft_len = 2*(no_bins-1)
# window = np.hamming(frame_size)
# stft_size = frame_size
# stft_shift = fft_len
# Y_f = np.zeros((30,467,32))
# for sig in Y:
#     #S_y = scipy.signal.stft(sig, nperseg=64,fs=fs_target, noverlap=0)
#     #S_y = stft(sig, fft_length=512, frame_shift=6000, window_length=512, window='hamming')
#
#     #S_y = specgram(sig, NFFT=62, Fs=fs_target, noverlap=0)
#     break
# #


no_bins = 32

cluster_bin = np.zeros((10, 10, no_bins))
d_vec = np.zeros(no_bins)
count = 0
with tqdm(total=no_bins) as pbar:
    for b in range(no_bins):  # (no_bins):
        Y_cell = []
        for p in range(0, 30, 3):
            Y_cell.append(Y_f_zm[p:p + 3, :, b])

        struc_bin, z_bin, d_hat_i = estimator.run(Y_cell, disp_struc=False)
        #print("d_hat= ",d_hat_i)
        d_vec[b] = d_hat_i
        cluster_bin[:d_hat_i, :, b] = z_bin
        count += 1
        pbar.update(1)


d_hat = int(np.round(np.mean(d_vec)))
vec_count = np.zeros(1025)
# computer majority vote:
for b in range(no_bins):
    for i in range(d_hat):
        arr= cluster_bin[i, :, b]
        idx = bool_to_int(arr.astype(np.int32))
        vec_count[idx] += 1


# pick the top d_hat after arg sorting in descending:
ranked = np.argsort(vec_count)
largest_ids = ranked[::-1][:5]
print(largest_ids)

print("experiment complete")
