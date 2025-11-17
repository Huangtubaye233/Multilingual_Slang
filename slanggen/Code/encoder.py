# Copyright (C) 2021 Zhewei Sun

import io
import abc
import pickle
import torch
import torch.nn as nn

import numpy as np

from .util import *

from sentence_transformers import SentenceTransformer

class WordEncoder:
    
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def embed_word(self, word):
        raise NotImplementedError()
        
    def norm_embed(self, word):
        vec = self.embed_word(word)
        return vec / np.linalg.norm(vec)
    
class FTEncoder(WordEncoder):
    
    def __init__(self, embed_file_name):
        fin = io.open(embed_file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        self.embeddings = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            self.embeddings[tokens[0]] = np.asarray(tokens[1:], dtype=np.float)
        
        self.vocab = set(self.embeddings.keys())
        self.E = self.embeddings[list(self.embeddings.keys())[0]].shape[0]
        
        self.cache = set()
    
    def embed_word(self, word):
        self.cache.add(word)
        return self.embeddings[word]
    
    def cache_embed(self, path):
        output = {}
        for word in self.cache:
            output[word] = self.embeddings[word]
        pickle.dump(output, open(path, 'wb'))
        
    def clear_cache(self):
        self.cache = set()
        
class FTCachedEncoder(WordEncoder):
    
    def __init__(self, embed_file_name):
        self.embeddings = pickle.load(open(embed_file_name, 'rb'))
        
        self.vocab = set(self.embeddings.keys())
        self.E = self.embeddings[list(self.embeddings.keys())[0]].shape[0]
        
    def embed_word(self, word):
        return self.embeddings[word]

class SenseEncoder:
    
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()
    
    def encode_dataset(self, dataset, slang_ind):
        
        embeds = {}
        
        def collect_slang_sents(dataset, ind):
            sentences = []
            for i in ind:
                text = dataset.slang_data[i].def_sent
                if contains_english(text):
                    sentences.append(' '.join(simple_preprocess(text)))
                else:
                    sentences.append(text)
            return sentences
        
        embeds['train'] = self.encode_sentences(collect_slang_sents(dataset, slang_ind.train))
        embeds['dev'] = self.encode_sentences(collect_slang_sents(dataset, slang_ind.dev))
        embeds['test'] = self.encode_sentences(collect_slang_sents(dataset, slang_ind.test))

        sentences = []
        for i in range(len(dataset.vocab)):
            word = dataset.vocab[i]
            for d in dataset.conv_data[word].definitions:
                text = d['def']
                if contains_english(text):
                    sentences.append(' '.join(simple_preprocess(text)))
                else:
                    sentences.append(text)
          
        embeds['standard'] = self.encode_sentences(sentences)
        
        return embeds
    
    @abc.abstractmethod
    def encode_sentences(self, sentences):
        raise NotImplementedError()
        
class SBertEncoder(SenseEncoder):
    
    def __init__(self, sbert_model_name=None, name=None):
        
        if sbert_model_name is None:
            sbert_model_name = 'bert-base-nli-mean-tokens'
            self.name = 'sbert_base'
        elif name is not None:
            self.name = name
        else:
            self.name = sbert_model_name
            
        self.sbert_model = SentenceTransformer(sbert_model_name)
        
    def encode_sentences(self, sentences):
        
        sbert_embeddings = np.asarray(self.sbert_model.encode(sentences))
        return normalize_L2(sbert_embeddings, axis=1)
    
class SBertWithHeadEncoder(SenseEncoder):
    def __init__(self, sbert_model_name, head_path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU cache before loading model")
        
        # Load SBERT
        self.sbert_model = SentenceTransformer(sbert_model_name).to(device)
        
        # Load head and config with memory optimization
        try:
            # Try loading with CPU first to avoid GPU memory issues
            checkpoint = torch.load(head_path, map_location='cpu')
            embed_dim = checkpoint['embed_dim']
            head_dim = checkpoint['head_dim']
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying with memory optimization settings...")
            # Set memory optimization
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
            checkpoint = torch.load(head_path, map_location=device)
            embed_dim = checkpoint['embed_dim']
            head_dim = checkpoint['head_dim']
        
        class TripletHead(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
            def forward(self, x):
                return self.linear(x)
        
        self.triplet_head = TripletHead(embed_dim, head_dim).to(device)

        # Load weights: support both finetuned schema (se_model/triplet_head)
        # and distilled schema (student_model/student_head)
        if 'se_model' in checkpoint and 'triplet_head' in checkpoint:
            self.sbert_model.load_state_dict(checkpoint['se_model'])
            self.triplet_head.load_state_dict(checkpoint['triplet_head'])
        elif 'student_model' in checkpoint and 'student_head' in checkpoint:
            self.sbert_model.load_state_dict(checkpoint['student_model'])
            self.triplet_head.load_state_dict(checkpoint['student_head'])
        else:
            raise KeyError('Unsupported checkpoint schema: expected (se_model, triplet_head) or (student_model, student_head)')
        self.sbert_model.eval()
        self.triplet_head.eval()
        self.name = head_path.split('/')[-1].replace('.pt','')

    def encode_sentences(self, sentences, batch_size=32):
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                # Encode with SBERT
                sbert_emb = self.sbert_model.encode(batch, convert_to_tensor=True).to(self.device)
                # Pass through head
                head_emb = self.triplet_head(sbert_emb)
                all_embeds.append(head_emb.cpu().numpy())
        return normalize_L2(np.vstack(all_embeds), axis=1)
    
def dump_vanilla_embeddings(trainer, slang_inds, fold_name, embed_name='SBERT_mpnet'):
    if embed_name == 'SBERT_mpnet':
        base_model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    elif embed_name == 'SBERT_LaBSE':
        base_model = 'sentence-transformers/LaBSE'
    elif embed_name == 'SBERT_e5_base':
        base_model = 'intfloat/multilingual-e5-base'
    elif embed_name == 'SBERT_e5_large':
        base_model = 'intfloat/multilingual-e5-large'
    else:
        raise ValueError(f'Unknown embed_name: {embed_name}')

    enc = SBertEncoder(base_model)

    ds = trainer.dataset

    def collect_slang_sents(ind):
        sentences = []
        for i in ind:
            text = ds.slang_data[i].def_sent
            if contains_english(text):
                sentences.append(' '.join(simple_preprocess(text)))
            else:
                sentences.append(text)
        return sentences

    train_sents = collect_slang_sents(slang_inds.train)
    dev_sents   = collect_slang_sents(slang_inds.dev)
    test_sents  = collect_slang_sents(slang_inds.test)

    std_sents = []
    for i in range(ds.V):
        w = ds.vocab[i]
        for d in ds.conv_data[w].definitions:
            text = d['def']
            if contains_english(text):
                std_sents.append(' '.join(simple_preprocess(text)))
            else:
                std_sents.append(text)

    train_emb = enc.encode_sentences(train_sents)
    dev_emb   = enc.encode_sentences(dev_sents)
    test_emb  = enc.encode_sentences(test_sents)
    std_emb   = enc.encode_sentences(std_sents)

    out_dir = os.path.join(trainer.out_dir, fold_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'sum_embed_{embed_name}_with_head.npz')
    np.savez(out_path, train=train_emb, dev=dev_emb, test=test_emb, standard=std_emb)
    print(f'[Vanilla baseline saved] {out_path}')

    # Also save a compatible .pt checkpoint with an identity head so that
    # it can be loaded by SBertWithHeadEncoder for reproducibility.
    sb_dir = os.path.join(trainer.out_dir, fold_name, 'SBERT_data')
    os.makedirs(sb_dir, exist_ok=True)

    embed_dim = enc.sbert_model.get_sentence_embedding_dimension()
    head_dim = embed_dim

    class IdentityHead(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.linear = nn.Linear(d, d)
        def forward(self, x):
            return self.linear(x)

    head = IdentityHead(embed_dim)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(embed_dim))
        head.linear.bias.zero_()

    ckpt = {
        'se_model': enc.sbert_model.state_dict(),
        'triplet_head': head.state_dict(),
        'embed_dim': embed_dim,
        'head_dim': head_dim
    }
    pt_path = os.path.join(sb_dir, f'{embed_name}_with_head.pt')
    torch.save(ckpt, pt_path)
    print(f'[Vanilla checkpoint saved] {pt_path}')