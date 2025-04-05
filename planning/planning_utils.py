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


def get_random_stimuli(n=3):
    r = RandomWord()
    adjectives = [r.word(include_parts_of_speech=["adjectives"]).replace(" ", "_") for _ in range(n)]
    nouns = [r.word(include_parts_of_speech=["nouns"]).replace(" ", "_") for _ in range(n)]

    stimuli = []
    for i, noun in enumerate(nouns):
        for adjective in adjectives:
            stimuli.append(f"{adjective} {noun}")

    return stimuli, nouns, adjectives

def get_stimuli():
    stimuli = ["red animal", 
               "green animal", 
               "yellow animal", 
               "red vehicle", 
               "green vehicle", 
               "yellow vehicle", 
               "red fruit", 
               "green fruit", 
               "yellow fruit"]
    
    objects = [word.split()[1] for word in stimuli]
    colours = list(set([word.split()[0] for word in stimuli]))
    return stimuli, objects, colours

def shuffle_stimuli(stimuli):
    random.shuffle(stimuli)
    return stimuli

def get_reward(stimuli, start, stop, reward):
    """Predict reward points for a sequence of stimuli.
    
    Args:
        stimuli (list): List of stimuli in random order.
        start (str): Object at which the sequence starts.
        stop (str): Colour at which the sequence ends.
        reward (str): Category of object that brings 2 points of reward.

    Returns:
        list of str: Reward points descriptions for the sequence.
    """
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
                points.append(f"{stim} (2)")
            else:
                points.append(f"{stim} (-1)")
    
            if colour == stop:
                break

        index += 1  # move to the next stimulus

    return points

def get_accept_reject_choice(seq):
    seq = seq[seq.index('SEQUENCE:') + len('SEQUENCE:'):]
    if 'START' in seq:
        seq = seq[0:seq.index('START')]
    print(seq)
    try:
        numbers = re.findall(r'\([A-Za-z0-9_-]+\)', seq)
        numbers = [int(num.replace('(', '').replace(')', '')) for num in numbers]
        print(numbers)
    except Exception as e:
        "Couldn't convert to int, setting list to []"
        numbers = []
    if sum(numbers) > 0:
        return 1
    elif sum(numbers) < 0:
        return 0
    elif sum(numbers) == 0:
        return 0.5

def test_revaluation(seqs, model):
    result_bools = []
    result_preds = []
    for seq in seqs:
        print("Get true accept / reject:")
        true_a_v_r = get_accept_reject_choice(seq)
        print(true_a_v_r)
        input_str = seq[0:seq.index('SEQUENCE:') + len('SEQUENCE:')]
        continuation = model.continue_input(input_str, do_sample=False)
        print("Get pred accept / reject:")
        pred_a_v_r = get_accept_reject_choice(continuation)
        print(pred_a_v_r)
        result_preds.append(pred_a_v_r)
        if true_a_v_r == pred_a_v_r:
            print("match")
            result_bools.append(1)
        else:
            print("no match")
            result_bools.append(0)
    return (result_bools, result_preds)

def get_len(seq):
    return seq.count(',') - 2

def get_mean(val):
    return np.mean(val[0])

def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

def test_consolidation(base_dir, task_type='transition'):
    print(task_type)
    print("------------------------")
    checkpoints = glob.glob(os.path.join(base_dir, 'checkpoint*'))
    model_dirs = ['rule_model'] + checkpoints 
    print(model_dirs)

    with open(os.path.join(base_dir, f'test_{task_type}.txt')) as f:
        seqs = f.readlines()
        seqs = [s.replace('\n', '') for s in seqs]
        seqs = sorted(list(set(seqs)))

    results = {}
    for model_dir in model_dirs:
        print(model_dir)
        model = GPT(base_model=model_dir)
        results[os.path.basename(model_dir)] = test_revaluation(seqs, model)
    return results

def get_data_from_file(file_name):
    with open('data/' + file_name, 'rb') as f:
        all_results = pickle.load(f)

    df = pd.DataFrame(all_results)
    df = df.dropna(axis='columns')

    for col in df.columns:
        df[col] = df[col].apply(get_mean)

    stats_to_plot = df.describe()
    
    return stats_to_plot

def get_step_num(string):
    # Regular expression to match 'checkpoint-' followed by one or more digits
    match = re.match(r'checkpoint-(\d+)', string)
    
    # If there's a match and it's at the start of the string
    if match and match.start() == 0:
        return int(match.group(1))
    else:
        return 0
    
def produce_plot(description_string='-'):
    files = ['results_reward.pkl', 'results_transition.pkl', 'results_both.pkl']
    labels = ['Reward revaluation', 'Transition revaluation', 'Both revaluation']
    
    plt.figure(figsize=(3, 2.5))
    
    for ind, file in enumerate(files):
        stats_to_plot = get_data_from_file(file) 
        cols = stats_to_plot.columns.tolist()
        
        xs = [get_step_num(c) for c in cols]
        ys = [stats_to_plot[c]['mean'] for c in cols]
        yerrs = [stats_to_plot[c]['std'] / math.sqrt(stats_to_plot[c]['count']) for c in cols] 
        sorted_pairs = sorted(zip(xs, ys, yerrs), key=lambda x: x[0])  # Sort by xs
        sorted_xs, sorted_ys, sorted_yerrs = zip(*sorted_pairs)
        
        # Use errorbar instead of plot to include error margins
        plt.errorbar(sorted_xs, 
                     sorted_ys, 
                     yerr=sorted_yerrs,  # Specify the error bars
                     marker='o', 
                     capsize=5,
                     label=labels[ind])
    
    #plt.yticks(np.linspace(0, 1, num=11))
    plt.ylim(0,1)
    plt.ylabel('Accept / reject accuracy')
    plt.xlabel('Training steps')
    plt.legend()
    plt.savefig(f'revaluation over time{description_string}.png', dpi=500, bbox_inches='tight')
    plt.show()

