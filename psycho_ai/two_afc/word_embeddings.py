__all__ = ['pse', 'get_glove_100d']

import os 
import wget
import gzip
import numpy as np
from tqdm import tqdm


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

def pse(pairs, target, embeddings_dict):
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

    R, C = handpicked_pse.shape
    handpicked_pse['PSE'] = None
    for i in range(R):
        n = handpicked_pse.loc[i]['target']
        l = handpicked_pse.loc[i]['Left']  
        r = handpicked_pse.loc[i]['Right']
        handpicked_pse.loc[i, 'PSE'] = int(100*bisect_search(l, r, n, embeddings_dict, delta_alpha=1/100))

    handpicked_pse['PSE'] = handpicked_pse['PSE'].astype('float')
    handpicked_pse = handpicked_pse.groupby('target')['PSE'].mean().sort_values()

    plt.figure(figsize = (10, 15), dpi=300)
    plt.plot(handpicked_pse.values, handpicked_pse.index.values, linewidth = 2, c = 'b')
    plt.gca().tick_params(axis='both', which='major', labelsize=7)
    plt.xticks([0, 50, 100], ['100% Left Attribute', '50% Left Attribute \n 50% Right Attribute', '100% Right Attribute']);
    # plt.yticks([0, 0.5, 1], ['No', '', 'Yes']);
    plt.xlabel('PSE', fontsize=10);
    plt.ylabel('target', fontsize=10);
    # plt.title('Is it a male plumber?', fontsize=14);
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.grid(alpha=0.3)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)