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

# one-time debug flag for get_conv_def
_GET_CONV_DEF_PRINTED = False

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
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        # fallback for lists
        return np.asarray(x)

    va = to_numpy(a).astype(np.float32).ravel()
    vb = to_numpy(b).astype(np.float32).ravel()

    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    denom = (na * nb) + 1e-12
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)

    # if isinstance(a, np.ndarray):
    #         a = torch.tensor(a)
    #     if isinstance(b, np.ndarray):
    #         b = torch.tensor(b)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     a = a.to(device)
    #     b = b.to(device)

def encode_list(texts, model):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

def get_conv_def(word, conv_dataset):
    if word in conv_dataset.data:
        defs = conv_dataset.data[word].definitions
        if defs:
            # Return the definition string instead of the raw dict
            first_def = defs[0]
            global _GET_CONV_DEF_PRINTED
            if not _GET_CONV_DEF_PRINTED:
                try:
                    print("[get_conv_def debug] raw_first_def:", first_def)
                    sample_extracted = first_def.get('def', None) if isinstance(first_def, dict) else (first_def if isinstance(first_def, str) else None)
                    print("[get_conv_def debug] extracted_def:", sample_extracted)
                except Exception as e:
                    print("[get_conv_def debug] failed to print:", e)
                _GET_CONV_DEF_PRINTED = True
            if isinstance(first_def, dict):
                return first_def.get('def', None)
            # Fallback: if data already stored as string
            if isinstance(first_def, str):
                return first_def
    return None

def avg_slang_conv_sim(slang_ds, conv_ds, model, idx_subset=None, compute_margin=True, agg='mean'):
    if idx_subset is None:
        idx_subset = range(len(slang_ds.slang_data))

    # Legacy path: pairwise first-def only
    if not compute_margin:
        slang_txts, conv_txts = [], []
        for i in idx_subset:
            entry = slang_ds.slang_data[i]
            conv_def = get_conv_def(entry.word, conv_ds)
            if conv_def:
                slang_txts.append(entry.def_sent)
                conv_txts.append(conv_def)
        if len(slang_txts) == 0:
            return float('nan')
        slang_emb = encode_list(slang_txts, model)
        conv_emb  = encode_list(conv_txts, model)
        sims = [cos_sim(slang_emb[i], conv_emb[i]) for i in range(len(slang_txts))]
        return np.mean(sims)

    # Margin path: best-of-def for true word vs best-of-def among all other words
    # 1) Prepare all conventional definitions once
    conv_texts_all = []
    word_to_indices = {}
    for w, wobj in conv_ds.data.items():
        start = len(conv_texts_all)
        defs = wobj.definitions
        if len(defs) == 0:
            word_to_indices[w] = []
            continue
        for d in defs:
            conv_texts_all.append(d['def'] if isinstance(d, dict) else str(d))
        end = len(conv_texts_all)
        word_to_indices[w] = list(range(start, end))

    if len(conv_texts_all) == 0:
        return {'mean_true': float('nan'), 'mean_imp': float('nan'), 'mean_margin': float('nan')}

    conv_emb_all = encode_list(conv_texts_all, model)
    if isinstance(conv_emb_all, torch.Tensor):
        conv_mat = conv_emb_all.detach().cpu().numpy().astype(np.float32)
    else:
        conv_mat = np.asarray(conv_emb_all).astype(np.float32)

    # 2) Encode slang texts in one batch
    slang_txts = [slang_ds.slang_data[i].def_sent for i in idx_subset]
    slang_emb = encode_list(slang_txts, model)
    if isinstance(slang_emb, torch.Tensor):
        slang_mat = slang_emb.detach().cpu().numpy().astype(np.float32)
    else:
        slang_mat = np.asarray(slang_emb).astype(np.float32)

    # 3) Compute per-sample true/imp with per-word aggregation
    #    Aggregation policy controlled by `agg`: 'mean' or 'max'.
    #    true = per-word agg over its definitions
    #    imp  = for each OTHER word, per-word agg, then take word-level max
    trues, imps = [], []
    for local_idx, i in enumerate(idx_subset):
        word = slang_ds.slang_data[i].word
        true_inds = word_to_indices.get(word, [])
        if not true_inds:
            continue  # skip if no conventional defs for this word
        sim_all = slang_mat[local_idx].dot(conv_mat.T)

        # true per-word aggregation
        if agg == 'max':
            true_val = float(np.max(sim_all[true_inds]))
        else:
            true_val = float(np.mean(sim_all[true_inds]))

        # imp per-word aggregation, then word-level max
        per_word_vals = []
        for w_other, inds in word_to_indices.items():
            if w_other == word or len(inds) == 0:
                continue
            if agg == 'max':
                per_word_vals.append(float(np.max(sim_all[inds])))
            else:
                per_word_vals.append(float(np.mean(sim_all[inds])))
        imp_val = float(np.max(per_word_vals)) if per_word_vals else -1e9

        trues.append(true_val)
        imps.append(imp_val)

    if len(trues) == 0:
        return {'mean_true': float('nan'), 'mean_imp': float('nan'), 'mean_margin': float('nan')}
    trues = np.asarray(trues)
    imps = np.asarray(imps)
    margins = trues - imps
    return {'mean_true': float(np.mean(trues)), 'mean_imp': float(np.mean(imps)), 'mean_margin': float(np.mean(margins))}

def get_finetuned_slang_emb(slang_ds, model_obj, fold_name, params):
    slang_txts = [e.def_sent for e in slang_ds.slang_data]
    return model_obj.trainer.get_testtime_embeddings(
        slang_txts, fold_name=fold_name, model_path=params["embed_name"]
    )

def avg_slang_conv_sim_after(slang_ds, conv_ds, sl_embeds, model_obj, fold_name, params, idx_subset=None, compute_margin=True, agg=None):
    if idx_subset is None:
        idx_subset = range(len(slang_ds.slang_data))

    # allow params override
    if agg is None:
        agg = params.get('sim_agg', 'mean')

    # Legacy path: pairwise first-def only (using finetuned encoder)
    if not compute_margin:
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

    # Margin path: best-of-def for true word vs best-of-def among all other words (finetuned encoder)
    # Ensure slang embeddings tensor
    if isinstance(sl_embeds, np.ndarray):
        sl_mat = sl_embeds.astype(np.float32)
    elif isinstance(sl_embeds, list):
        sl_mat = torch.stack(sl_embeds).cpu().numpy().astype(np.float32)
    elif isinstance(sl_embeds, torch.Tensor):
        sl_mat = sl_embeds.detach().cpu().numpy().astype(np.float32)
    else:
        raise TypeError('Unsupported sl_embeds type')

    # 1) Prepare all conventional definitions once
    conv_texts_all = []
    word_to_indices = {}
    for w, wobj in conv_ds.data.items():
        start = len(conv_texts_all)
        defs = wobj.definitions
        if len(defs) == 0:
            word_to_indices[w] = []
            continue
        for d in defs:
            conv_texts_all.append(d['def'] if isinstance(d, dict) else str(d))
        end = len(conv_texts_all)
        word_to_indices[w] = list(range(start, end))

    if len(conv_texts_all) == 0:
        return {'mean_true': float('nan'), 'mean_imp': float('nan'), 'mean_margin': float('nan')}

    conv_emb_all = model_obj.trainer.get_testtime_embeddings(
        conv_texts_all, fold_name=fold_name, model_path=params["embed_name"]
    )
    if isinstance(conv_emb_all, torch.Tensor):
        conv_mat = conv_emb_all.detach().cpu().numpy().astype(np.float32)
    else:
        conv_mat = np.asarray(conv_emb_all).astype(np.float32)

    # Restrict to subset order
    subset_indices = list(idx_subset)
    slang_mat = sl_mat[subset_indices]

    trues, imps = [], []
    for local_pos, i in enumerate(subset_indices):
        word = slang_ds.slang_data[i].word
        true_inds = word_to_indices.get(word, [])
        if not true_inds:
            continue
        sim_all = slang_mat[local_pos].dot(conv_mat.T)

        # true per-word aggregation
        if agg == 'max':
            true_val = float(np.max(sim_all[true_inds]))
        else:
            true_val = float(np.mean(sim_all[true_inds]))

        # imp per-word aggregation, then word-level max
        per_word_vals = []
        for w_other, inds in word_to_indices.items():
            if w_other == word or len(inds) == 0:
                continue
            if agg == 'max':
                per_word_vals.append(float(np.max(sim_all[inds])))
            else:
                per_word_vals.append(float(np.mean(sim_all[inds])))
        imp_val = float(np.max(per_word_vals)) if per_word_vals else -1e9

        trues.append(true_val)
        imps.append(imp_val)

    if len(trues) == 0:
        return {'mean_true': float('nan'), 'mean_imp': float('nan'), 'mean_margin': float('nan')}
    trues = np.asarray(trues)
    imps = np.asarray(imps)
    margins = trues - imps
    return {'mean_true': float(np.mean(trues)), 'mean_imp': float(np.mean(imps)), 'mean_margin': float(np.mean(margins))}
