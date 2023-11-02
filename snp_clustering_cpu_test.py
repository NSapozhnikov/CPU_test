#!/usr/bin/env python3
"""
Clusterization of LD matrices with dbscan and hdsbcan
Author: Nikita Sapozhnikov, nikita.sapozhnikov1@gmail.com
Date: November 02, 2023
snp_clustering for CPU time tests
"""

import time
import os
import sys
import numpy as np
from sklearn.cluster import DBSCAN
import h5py


EPS = np.linspace(0.05, 1, 20)
MIN_SAMPLES = np.arange(2, 20, 1)


def prepare_data() -> np.ndarray:
    """
    data preparation function
    """
    matrix_file_path = os.path.join('data', 'chr3-1.ld.h5')
    snp_file_path = os.path.join('data', 'chr3-1.snplist')
    try:
        print('Openning snplist file...')
        with open(snp_file_path, 'r', encoding='utf-8') as snp_file:
            snp_list = np.loadtxt(snp_file, dtype=str)
    except FileNotFoundError:
        sys.exit('Invalid snplist file path.')
    try:
        print('Openning matrix file...')
        with h5py.File(matrix_file_path, 'r') as corr_file:
            dset = corr_file['r2']
            ld_matrix = dset['block0_values']
            corr_matrix = np.array(ld_matrix)
    except FileNotFoundError:
        sys.exit('Invalid matrix file path.')

    np.nan_to_num(corr_matrix, copy=False)
    diss_matrix = 1 - np.abs(corr_matrix, out=corr_matrix)
    diss_matrix = diss_matrix.reshape(len(snp_list), len(snp_list))
    np.fill_diagonal(diss_matrix, 0)
    print(diss_matrix)
    return diss_matrix


def dbscan_clustering(diss_matrix: np.ndarray,
                      eps: float,
                      min_samples: int) -> float:
    """
    perform a dbscan clustering
    """
    time_start = time.time()
    db = DBSCAN(eps=eps,
                min_samples=min_samples,
                metric='precomputed',
                n_jobs=-1).fit(diss_matrix)
    time_end = time.time()
    clusterization_time = time_end - time_start
    print('Time of clustering: ', clusterization_time)
    return clusterization_time


if __name__ == '__main__':
    time_list = []
    matrix = prepare_data()
    for eps_ in EPS:
        for min_samples_ in MIN_SAMPLES:
            print(f'Parameters pair:\neps:\t{eps_}\nmin_samples:\t{min_samples_}')
            iter_time = dbscan_clustering(diss_matrix=matrix,
                                          eps=eps_,
                                          min_samples=min_samples_)
            time_list.append(iter_time)
            print('Total time is: ', sum(time_list))
