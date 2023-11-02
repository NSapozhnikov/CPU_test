# CPU_test

## Overview
A python code to compare different setups' performance on solving clusterization task with DBSCAN algorithm.

## Usage
1. Data can be downloaded from [here](https://drive.google.com/drive/folders/144dQUME9Qw6y8UqP4hXozyHf08qiROex?usp=drive_link). Put the data directory and a script in the same directory.
2. Use `python -m venv venv` to create a virtual enviroment. Use `venv\Scripts\activate` (Win) or `source venv/bin/activate` (Unix) to activate the enviroment.
3. Use `pip install -r requirements.txt` to install all dependency Python modules (for the full list see the requirements.txt).
4. Run `python snp_clustering_cpu_test.py`

The script will exctract and prepare the data. And then it will start clusterization with a parameter grid of 20x20. You will be able to see both 1 instance of clusterization and tottal time for all parameter pairs clusterizations.
