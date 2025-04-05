import sys
sys.path.append('../scripts')

import random
import pandas as pd
import logging
from random import shuffle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
import string
import os
import re
import glob
import torch
from wonderwords import RandomWord
import gc
import pickle
from sklearn.linear_model import LinearRegression
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import pearsonr
import math
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def get_accept_reject(stimuli, start, stop, reward):

    points = []
    sequence_started = False
    
    index = 0  # start index
    while True:
        stim = stimuli[index % len(stimuli)]
        colour, obj = stim.split()

        if not sequence_started:
            if stim == start:
                sequence_started = True
            index += 1
            continue
        else:
            if obj == reward:
                points.append(2)
            else:
                points.append(-1)
    
            if colour == stop:
                break

        index += 1  # move to the next stimulus

    print(f"Inferred points sequence for {start} / {stop} / {reward}: {points}")
    if sum(points) > 0:
        return 1
    if sum(points) < 0:
        return 0
    if sum(points) ==0:
        return 0.5

def get_pred_reward_for_strategy(seq, output_dir, strategy):
    with open(os.path.join(output_dir, 'trial_info.pkl'), 'rb') as handle:
        trial_info = pickle.load(handle)
        stimuli = trial_info['stimuli']
        train_stop = trial_info['train_stop']
        train_reward = trial_info['train_reward']
    
    test_start = seq[0:seq.index(', STOP')].replace('START: ', '')
    test_stop = seq[seq.index('STOP'):seq.index('REWARD')].replace('STOP: ', '').replace(', ', '')
    test_reward = seq[seq.index('REWARD'):seq.index('SEQUENCE')].replace('REWARD: ', '').replace(', ', '')
    
    print(f'Train stop: {train_stop}, train reward: {train_reward}, test stop: {test_stop}, test reward: {test_reward}')
    print(stimuli)
    
    if strategy == 'revaluate_reward':
        # get train stop, test reward
        outcome = get_accept_reject(stimuli, test_start, train_stop, test_reward)
    if strategy == 'revaluate_transition':
        # get train reward, test stop
        outcome = get_accept_reject(stimuli, test_start, test_stop, train_reward)
    if strategy == 'revaluate_both':
        # get test reward, test stop
        outcome = get_accept_reject(stimuli, test_start, test_stop, test_reward)
    if strategy == 'no_revaluation':
        # get train reward, train stop
        outcome = get_accept_reject(stimuli, test_start, train_stop, train_reward)
    return outcome

def get_strategy(i, task_type='reward', strategy='no_revaluation'):
    model_dir = f'./clm_script_{i}'
    
    with open(os.path.join(model_dir, f'test_{task_type}.txt')) as f:
        seqs = f.readlines()
        print(f"{len(seqs)} seqs.")
        seqs = [s.replace('\n', '') for s in seqs]
        seqs = sorted(list(set(seqs)))
        print(task_type, strategy)
    
    preds = []
    for seq in seqs:
        pred = get_pred_reward_for_strategy(seq, model_dir, strategy)
        preds.append(pred)
        print(seq)
        print(f"Prediction: {pred}")
        print("......................")
    return np.asarray(preds)

def get_strategy_df(task_type='transition'):
    file_name = f'data/results_{task_type}.pkl'
    with open(file_name, 'rb') as f:
        all_results = pickle.load(f)
    df = pd.DataFrame(all_results)
    df = df.dropna(axis='columns')

    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x[1]))

    df["model_num"] = df.index
    print(df.index.tolist())

    df['NR_preds'] = df["model_num"].apply(lambda x: get_strategy(x, 
                                                                  task_type=task_type, 
                                                                  strategy='no_revaluation'))

    df['RR_preds'] = df["model_num"].apply(lambda x: get_strategy(x, 
                                                                  task_type=task_type, 
                                                                  strategy='revaluate_reward'))

    df['TR_preds'] = df["model_num"].apply(lambda x: get_strategy(x, 
                                                                  task_type=task_type, 
                                                                  strategy='revaluate_transition'))

    df['BR_preds'] = df["model_num"].apply(lambda x: get_strategy(x, 
                                                                  task_type=task_type, 
                                                                  strategy='revaluate_both'))
    return df