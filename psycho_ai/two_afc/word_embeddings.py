__all__ = ['get_glove_100d', 'pse', 'jnd', 'plot_pse', 'bisect_search', 'twoafc_experiment', 'similarity', 'psy_cur']

import os 
import wget
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cosine
import re

def similarity(x, y):
    return 1 - cosine(x, y)

def get_glove_100d():
    if 'glove.6B.100d.txt.gz' not in os.listdir():
        wget.download('https://github.com/allenai/spv2/raw/master/model/glove.6B.100d.txt.gz', './glove.6B.100d.txt.gz')
    embeddings_dict = {}
    with gzip.open('glove.6B.100d.txt.gz', 'r') as f:
        for line in tqdm(f):
            values = line.decode().split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# def pse(target_words, left_words, right_words, embedding):
#     return

def twoafc_experiment(left_embedding, right_embedding, target_embeddings):
    decisions = []
    for embedding in target_embeddings:
        for idx in range(0, 100, 1):
            alpha = idx/100
            # woman*n*(1-alpha) man*n*alpha great ____
            left_score = (1-alpha) + \
                alpha*similarity(left_embedding, right_embedding) + \
                similarity(left_embedding, embedding)
            right_score = (1-alpha)*similarity(right_embedding, left_embedding) + \
                alpha + \
                + similarity(right_embedding, embedding)
            d = int(right_score > left_score)
            decisions.append([alpha, d])
    return pd.DataFrame(np.array(decisions), columns=['alpha', 'd'])
    
    
def bisect_search(left_word, right_word, target_word, model, delta_alpha=1/100):
    try:
        model[right_word]
    except KeyError:
        print(right_word + ' not in dictionary')
    try:
        model[left_word]
    except KeyError:
        print(left_word + ' not in dictionary')
    try:
        model[target_word]
    except KeyError:
        print(target_word + ' not in dictionary')
    left_token = model[left_word]
    right_roken = model[right_word]
    target_token = model[target_word]
    min_alpha = 0
    max_alpha = 1
    c = 0
    def eval_alpha(alpha):
        left_score = (1-alpha) + \
                alpha*similarity(left_token, right_roken) + \
                similarity(left_token, target_token)
        right_score = (1-alpha)*similarity(right_roken, left_token) + \
            alpha + \
            + similarity(right_roken, target_token)
        return left_score, right_score
    
    while (max_alpha-min_alpha) > delta_alpha:
        min_left_score, min_right_score = eval_alpha(min_alpha)
        min_alpha_d = int(min_right_score > min_left_score)
        max_left_score, max_right_score = eval_alpha(max_alpha)
        max_alpha_d = int(max_right_score > max_left_score)
        
        middle_left_score, middle_right_score = eval_alpha((min_alpha + max_alpha)/2)
        middle_alpha_d = int(middle_right_score > middle_left_score)        

        if middle_alpha_d != min_alpha_d:
            max_alpha = (min_alpha + max_alpha) / 2
        elif middle_alpha_d != max_alpha_d :
            min_alpha = (min_alpha + max_alpha) / 2
        else:
            break
        c += 1
        if c > 10:
            break
    return (min_alpha + max_alpha) / 2

def psy_cur(right_word, left_word, target_word, embeddings_dict):
    plt.figure(figsize=(10, 4))
    try:
        embeddings_dict[right_word]
    except KeyError:
        print(right_word + ' not in dictionary')
    try:
        embeddings_dict[left_word]
    except KeyError:
        print(left_word + ' not in dictionary')
    try:
        embeddings_dict[target_word]
    except KeyError:
        print(target_word + ' not in dictionary')
    d = twoafc_experiment(embeddings_dict[right_word], 
                      embeddings_dict[left_word],
                      [embeddings_dict[target_word]])
    plt.plot(d['alpha'], d['d'], 'b', alpha=1.0, linewidth=2);
    plt.gca().tick_params(axis='both', which='major', labelsize=7)
    plt.xticks([0, 0.5, 1], ['100% left word', '50% right\n50% left', '100% right word']);
    plt.yticks([0, 0.5, 1], ['Left', '', 'Right']);
    plt.xlabel('Stimulus', fontsize=10);
    plt.ylabel('Response', fontsize=10);
    plt.title('Psychometric Curve', fontsize=10);
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

def jnd(embedding_list, target, pairs):
    w_dict = {} 
    w_dict['target'] = []
    w_dict['Left'] = []
    w_dict['Right'] = []
    for n in target:
        for pair in pairs:
            w_dict['Left'].append(pair[0])
            w_dict['Right'].append(pair[1])
            w_dict['target'].append(n)
    handpicked_pse = pd.DataFrame(w_dict)
    
    pse_list = []
    for embedding in embedding_list:
        R, C = handpicked_pse.shape
        handpicked_pse['PSE'] = None
        for i in range(R):
            n = handpicked_pse.loc[i]['target']
            l = handpicked_pse.loc[i]['Left']  
            r = handpicked_pse.loc[i]['Right']
            handpicked_pse.loc[i, 'PSE'] = bisect_search(l, r, n, embedding, delta_alpha=1/100)

        handpicked_pse['PSE'] = handpicked_pse['PSE'].astype('float')
        pse_list.append(handpicked_pse.groupby('target')['PSE'].mean())
        handpicked_jnd = pd.concat(pse_list).groupby('target').var()
    return handpicked_jnd.sort_values().to_dict()

def pse(embeddings_dict, target_list, pairs):
    pse_dict = {}
    for phase in target_list:
        embedding_ = embeddings_dict
        w_list = re.split(r'\s+|-|_|\.',phase)
        w_list = list(filter(None, w_list))
        embedding_[phase] = np.mean([embedding_[x] for x in w_list], axis=0)
        pse_score = 0
        for p in pairs:
            l, r = p[0], p[1]
            pse_score += bisect_search(l, r, phase, embedding_, delta_alpha=1/100)
        pse = pse_score / len(pairs) 
        pse_dict[phase] = pse
    return {k: v for k, v in sorted(pse_dict.items(), key=lambda item: item[1])}


f = plt.figure(figsize = (6, 6), dpi=150)
def plot_pse(pse_score, fig=f):
    f, ax = fig
    ax.plot(list(pse_score.values()), list(pse_score.keys()), linewidth = 2, c = 'b')
    ax.gca().tick_params(axis='y', which='major', labelsize=12)
    ax.gca().tick_params(axis='x', which='major', direction="in", labelsize=8)

    ax.xticks([0, .5, 1], ['100% \n                 Female Attribute', '50% Female Attribute \n 50% Male Attribute', '100% \n Male Attribute              ']);
    # plt.yticks([0, 0.5, 1], ['No', '', 'Yes']);
    ax.xlabel('PSE', fontsize=15)
    ax.ylabel('Occupation', fontsize=15)
    # plt.title('Is it a male plumber?', fontsize=14);
    ax.gca().spines['right'].set_visible(False)
    ax.gca().spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.gca().yaxis.set_ticks_position('left')
    ax.gca().xaxis.set_ticks_position('bottom')
    ax.grid(alpha=0.3)
    ax.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return ax