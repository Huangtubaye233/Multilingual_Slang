# Copyright (C) 2021 Zhewei Sun

import os
import re
import shutil
from collections import namedtuple

import numpy as np
import torch

from nltk.corpus import stopwords as sw
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Helper functions

punctuations = '!"#$%&()\*\+,-\./:;<=>?@[\\]^_`{|}~'

re_punc = re.compile(r"["+punctuations+r"]+")
re_space = re.compile(r" +")

stopwords = set(sw.words('english'))

Definition = namedtuple('Definition', ['word', 'type', 'def_sent', 'meta_data'])
# For slang data entries
SlangEntry = namedtuple('SlangEntry', ['word', 'def_sent', 'meta_data'])
DataIndex = namedtuple('DataIndex', ['train', 'dev', 'test'])
Triplet = namedtuple('Triplet', ['anchor', 'positive', 'negative'])

def tokenize(sentence):
    return re.compile(r"(?:^|(?<=\s))\S+(?=\s|$)").findall(sentence)

def processTokens(fun, sentence):
    return re.compile(r"(?:^|(?<=\s))\S+(?=\s|$)").sub(fun, sentence)

def normalize(array, axis=1):
    denoms = np.sum(array, axis=axis)
    if axis == 1:
        return array / denoms[:,np.newaxis]
    if axis == 0:
        return array / denoms[np.newaxis, :]
    
def normalize_L2(array, axis=1):
    if axis == 1:
        return array / np.linalg.norm(array, axis=1)[:, np.newaxis]
    if axis == 0:
        return array / np.linalg.norm(array, axis=0)[np.newaxis, :]
    
def acronym_check(entry):
    if 'acronym' in entry.def_sent:
        return True
    for c in str(entry.word):
        if ord(c) >= 65 and ord(c) <= 90:
            continue
        return False
    return True

def is_close_def(query_sent, target_sent, threshold=0.5):
    # Only apply English token overlap when both sentences contain English letters.
    if contains_english(query_sent) and contains_english(target_sent):
        query_s = [w for w in simple_preprocess(query_sent) if w not in stopwords]
        target_s = set([w for w in simple_preprocess(target_sent) if w not in stopwords])
        overlap_c = 0
        for word in query_s:
            if word in target_s:
                overlap_c += 1
        return overlap_c >= len(query_s) * threshold
    # For non-English cases, skip overlap filtering to avoid false negatives.
    return False

def contains_english(text):
    """Return True if the text contains any ASCII English letters.

    This lightweight detector avoids extra dependencies and is sufficient
    for deciding whether English-oriented preprocessing should be applied.
    """
    return re.search(r"[A-Za-z]", str(text)) is not None

def has_close_conv_def(word, slang_def_sent, conv_data, threshold=0.5):
    conv_sents = [d['def'] for d in conv_data[word].definitions]
    for conv_sent in conv_sents:
        if is_close_def(slang_def_sent, conv_sent, threshold):
            return True
    return False

def create_directory(path):
    try: 
        # Normalize path to handle double slashes
        path = os.path.normpath(path)
        # Use makedirs instead of mkdir to create nested directories
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error while creating directory {path}: {e}")

# For conventional data entries
class Word:
    
    def __init__(self, word):
        self.word = word
        self.pos_tags = set()
        self.definitions = []

    def attach_def(self, word_def, pos, sentences):
        new_def = {'def':word_def, 'pos':pos, 'sents':sentences}
        self.pos_tags.add(pos)
        self.definitions.append(new_def)
        
# Evaluation helpers

def get_rankings(l_model, inds, labels):
    N = l_model.shape[0]
    ranks = np.zeros(l_model.shape, dtype=np.int32)
    rankings = np.zeros(N, dtype=np.int32)
        
    for i in range(N):
        ranks[i] = np.argsort(l_model[i])[::-1]
        rankings[i] = ranks[i].tolist().index(labels[inds[i]])+1
            
    return rankings
    
def get_roc(rankings, N_cat):
    roc = np.zeros(N_cat+1)
    for rank in rankings:
        roc[rank]+=1
    for i in range(1,N_cat+1):
        roc[i] = roc[i] + roc[i-1]
    return roc / rankings.shape[0]

# Multilingual helpers

def cos_sim(a, b):
    if isinstance(a, np.ndarray):
        a = torch.tensor(a)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = a.to(device)
    b = b.to(device)

    return float(util.cos_sim(a, b)[0][0])

def encode_list(texts, model):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

def get_conv_def(word, conv_dataset):
    if word in conv_dataset.data:
        defs = conv_dataset.data[word].definitions
        if defs:
            return defs[0]
    return None

def avg_slang_conv_sim(slang_ds, conv_ds, model, idx_subset=None):
    if idx_subset is None:
        idx_subset = range(len(slang_ds.slang_data))

    slang_txts, conv_txts = [], []
    for i in idx_subset:
        entry = slang_ds.slang_data[i]
        conv_def = get_conv_def(entry.word, conv_ds)
        if conv_def:
            slang_txts.append(entry.def_sent)
            conv_txts.append(conv_def)

    slang_emb = encode_list(slang_txts, model)
    conv_emb  = encode_list(conv_txts, model)

    sims = [cos_sim(slang_emb[i], conv_emb[i]) for i in range(len(slang_txts))]
    return np.mean(sims)

def get_finetuned_slang_emb(slang_ds, model_obj, fold_name, params):
    slang_txts = [e.def_sent for e in slang_ds.slang_data]
    return model_obj.trainer.get_testtime_embeddings(
        slang_txts, fold_name=fold_name, model_path=params["embed_name"]
    )

def avg_slang_conv_sim_after(slang_ds, conv_ds, sl_embeds, model_obj, fold_name, params, idx_subset=None):
    if idx_subset is None:
        idx_subset = range(len(slang_ds.slang_data))

    conv_txts, sims = [], []
    for local_idx in idx_subset:
        entry = slang_ds.slang_data[local_idx]
        conv_def = get_conv_def(entry.word, conv_ds)
        if conv_def:
            conv_txts.append(conv_def)
            sims.append(local_idx)

    conv_emb = model_obj.trainer.get_testtime_embeddings(
        conv_txts, fold_name=fold_name, model_path=params["embed_name"]
    )

    if isinstance(sl_embeds, np.ndarray):
        sl_embeds = torch.tensor(sl_embeds)
    elif isinstance(sl_embeds, list):
        sl_embeds = torch.stack(sl_embeds)

    if isinstance(conv_emb, np.ndarray):
        conv_emb = torch.tensor(conv_emb)
    elif isinstance(conv_emb, list):
        conv_emb = torch.stack(conv_emb)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sl_embeds = sl_embeds.to(device)
    conv_emb = conv_emb.to(device)

    sims = [cos_sim(sl_embeds[i], conv_emb[j]) for j, i in enumerate(sims)]
    return np.mean(sims)
