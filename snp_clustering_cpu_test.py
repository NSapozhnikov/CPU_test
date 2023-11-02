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
import resource
import numpy as np
from sklearn.cluster import DBSCAN
import h5py


EPS = np.linspace(0.05, 1, 20)
MIN_SAMPLES = np.arange(2, 20, 1)


def get_memory_usage():
    """
    get memory utilization
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return (usage.ru_maxrss / 1024.0)  # Convert to kilobytes


def prepare_data() -> np.ndarray:
    """
    data preparation function
    """
    matrix_file_path = os.path.join('data', 'chr3-1.ld.h5')
    try:
        print('Openning matrix file...')
        with h5py.File(matrix_file_path, 'r') as corr_file:
            dset = corr_file['r2']
            ld_matrix = dset['block0_values']
            corr_matrix = np.array(ld_matrix)
    except FileNotFoundError:
        sys.exit('Invalid matrix file path.')

    np.nan_to_num(corr_matrix, copy=False)
    np.abs(corr_matrix, out=corr_matrix)
    corr_matrix = 1 - corr_matrix
    np.fill_diagonal(corr_matrix, 0)
    print(corr_matrix)
    print(f"Memory usage: {get_memory_usage():.2f} KB")
    return corr_matrix


def dbscan_clustering(diss_matrix: np.ndarray,
                      eps: float,
                      min_samples: int) -> float:
    """
    perform a dbscan clustering
    """
    time_start = time.time()
    DBSCAN(eps=eps,
           min_samples=min_samples,
           metric='precomputed',
           n_jobs=-1).fit(diss_matrix)
    time_end = time.time()
    clusterization_time = time_end - time_start
    print('Time of clustering: ', clusterization_time)
    print(f"Memory usage: {get_memory_usage():.2f} KB")
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
        mean = np.mean(time_list)
        print(f'Mean for eps = {eps_}:', mean)

        median = np.median(time_list)
        print(f'Median for eps = {eps_}:', median)
        # ddof=1 for sample standard deviation
        std_dev = np.std(time_list, ddof=1)
        print(f'Standart Deviation for eps = {eps_}:', std_dev)
