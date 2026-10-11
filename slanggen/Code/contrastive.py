# Copyright (C) 2021 Zhewei Sun

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import scipy.spatial.distance as dist

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator

from tqdm import trange

from .util import *
from .encoder import SBertWithHeadEncoder
import os
from pathlib import Path
import torch.nn.functional as F


def read_triplet_csv(path):
    # Preserve literal definitions such as "20" and "NA" through the CSV round trip.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ('anchor', 'positive', 'negative'):
        for row, text in enumerate(frame[column]):
            try:
                prepare_definition_text(text)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid {column} at data row {row + 1}") from exc
    return frame

class SlangGenTrainer:
    
    MAX_NEIGHBOR = 300
    
    def __init__(self, dataset, word_encoder, out_dir='', verbose=False):
        
        self.out_dir = out_dir
        create_directory(out_dir)
        
        self.dataset = dataset
        
        self.word_encoder = word_encoder
        
        self.verbose = verbose
            
        conv_lens = []
        for i in range(dataset.V):
            word = dataset.vocab[i]
            conv_lens.append(len(dataset.conv_data[word].definitions))
        self.conv_lens = np.asarray(conv_lens)

        self.conv_acc = np.zeros(dataset.V, dtype=np.int32)

        for i in range(1,dataset.V):
            self.conv_acc[i] = self.conv_acc[i-1] + self.conv_lens[i-1]
            
        self.word_dist = self.preprocess_word_dist()
        np.save(out_dir+'/word_dist.npy', self.word_dist)
        
        self.sense_encoder = None
        self.se_model_name = "INVALID"

    def preprocess_slang_data(self, slang_ind, fold_name='default', skip_steps=[]):

        out_dir = self.out_dir + '/' + fold_name
        create_directory(out_dir)
        out_dir += '/'
        
        # Generate contrastive pairs for training
        if 'contrastive' not in skip_steps:
            if self.verbose:
                print("Generating contrative pairs...")
            contrastive_pairs_train, contrastive_pairs_dev = self.preprocess_contrastive(slang_ind)
            np.save(out_dir+'contrastive_train.npy', contrastive_pairs_train)
            np.save(out_dir+'contrastive_dev.npy', contrastive_pairs_dev)
            if self.verbose:
                print("Complete!")
                
    def load_preprocessed_data(self, fold_name='default', skip_steps=[]):
        
        out_dir = self.out_dir + '/' + fold_name + '/'
        
        preproc_data = {}
        
        if 'contrastive' not in skip_steps:
            preproc_data['cp_train'] = np.load(out_dir+'contrastive_train.npy', allow_pickle=True)
            preproc_data['cp_dev'] = np.load(out_dir+'contrastive_dev.npy', allow_pickle=True)
            
        return preproc_data
    
    # def load_sense_encoder(self, model_name, model_path):
        
    #     if self.se_model_name == model_name:
    #         return self.sense_encoder
        
    #     self.sense_encoder = SBertEncoder(sbert_model_name=model_name, name=model_path)
    #     self.se_model_name = model_name
        
    def load_sense_encoder(self, model_name, model_path):
        if self.se_model_name == model_name:
            return self.sense_encoder
        # Check if SE model exists
        SE_head_path = os.path.join(os.path.dirname(model_name), model_path + '_with_head.pt')
        SE_whole_path = os.path.join(os.path.dirname(model_name), model_path + '_whole_finetuned.pt')
        SE_multilingual_path = os.path.join(os.path.dirname(model_name), model_path + '_multilingual_distilled.pt')
        # SE_head_path = model_name + '_with_head.pt'
        
        # Check for multilingual distilled model first (now load directly)
        if os.path.exists(SE_multilingual_path):
            # Determine the base model from the saved checkpoint metadata
            checkpoint = torch.load(SE_multilingual_path, map_location='cpu')
            teacher_model_path = checkpoint.get('teacher_model_path', '')
            teacher_model_name = teacher_model_path.split('/')[-1]

            if 'mpnet' in teacher_model_name:
                base_model_name = 'paraphrase-multilingual-mpnet-base-v2'
            elif 'e5_base' in teacher_model_name:
                base_model_name = 'intfloat/multilingual-e5-base'
            elif 'e5_large' in teacher_model_name:
                base_model_name = 'intfloat/multilingual-e5-large'
            elif 'multilingual-MiniLM' in teacher_model_name:
                base_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            elif 'LaBSE' in teacher_model_name:
                base_model_name = 'LaBSE'
            else:
                # default to mpnet if teacher name unavailable
                base_model_name = 'paraphrase-multilingual-mpnet-base-v2'

            # Directly load distilled checkpoint with SBertWithHeadEncoder
            self.sense_encoder = SBertWithHeadEncoder(base_model_name, SE_multilingual_path)
        elif os.path.exists(SE_whole_path):
            if model_path == 'SBERT_contrastive':
                self.sense_encoder = SBertWithHeadEncoder('bert-base-nli-mean-tokens', SE_whole_path)
            elif model_path == 'SBERT_t5':
                self.sense_encoder = SBertWithHeadEncoder('sentence-t5-base', SE_whole_path)
            elif model_path == 'SBERT-multilingual-MiniLM-L12-v2':
                self.sense_encoder = SBertWithHeadEncoder('paraphrase-multilingual-MiniLM-L12-v2', SE_whole_path)
            elif model_path == 'SBERT_LaBSE':
                self.sense_encoder = SBertWithHeadEncoder('LaBSE', SE_whole_path)
            elif model_path == 'SBERT_mpnet':
                self.sense_encoder = SBertWithHeadEncoder('paraphrase-multilingual-mpnet-base-v2', SE_whole_path)
            elif model_path == 'SBERT_e5_base':
                self.sense_encoder = SBertWithHeadEncoder('intfloat/multilingual-e5-base', SE_whole_path)
            elif model_path == 'SBERT_e5_large':
                self.sense_encoder = SBertWithHeadEncoder('intfloat/multilingual-e5-large', SE_whole_path)
            else:
                raise ValueError(f"Unsupported model_path: {model_path}")
        elif os.path.exists(SE_head_path):
            if model_path == 'SBERT_contrastive':
                self.sense_encoder = SBertWithHeadEncoder('bert-base-nli-mean-tokens', SE_head_path)
            elif model_path == 'SBERT_t5':
                self.sense_encoder = SBertWithHeadEncoder('sentence-t5-base', SE_head_path)
            elif model_path == 'SBERT-multilingual-MiniLM-L12-v2':
                self.sense_encoder = SBertWithHeadEncoder('paraphrase-multilingual-MiniLM-L12-v2', SE_head_path)
            elif model_path == 'SBERT_LaBSE':
                self.sense_encoder = SBertWithHeadEncoder('LaBSE', SE_head_path)
            elif model_path == 'SBERT_mpnet':
                self.sense_encoder = SBertWithHeadEncoder('paraphrase-multilingual-mpnet-base-v2', SE_head_path)
            elif model_path == 'SBERT_e5_base':
                self.sense_encoder = SBertWithHeadEncoder('intfloat/multilingual-e5-base', SE_head_path)
            elif model_path == 'SBERT_e5_large':
                self.sense_encoder = SBertWithHeadEncoder('intfloat/multilingual-e5-large', SE_head_path)
            else:
                raise ValueError(f"Unsupported model_path: {model_path}")
        else:
            # No trained model found - this should not happen in normal workflow
            raise FileNotFoundError(f"No trained model found for {model_path}. Please run training first. Expected files: {SE_whole_path}, {SE_head_path}, or {SE_multilingual_path}")
        self.se_model_name = model_name
        
    
    def get_trained_embeddings(self, slang_ind, fold_name='default', model_path='SBERT_contrastive'):
        
        # Set up directory structure based on model_path
        if model_path == 'SBERT_contrastive' or model_path == 'SBERT_t5' or model_path == 'SBERT-multilingual-MiniLM-L12-v2' or model_path == 'SBERT_LaBSE' or model_path == 'SBERT_mpnet' or model_path == 'SBERT_e5_base' or model_path == 'SBERT_e5_large':
            model_name = self.out_dir + '/' + fold_name + '/SBERT_data/' + model_path
        else:
            raise ValueError(f"Unsupported embed_name: {model_path}")
            
        self.load_sense_encoder(model_name, model_path)
        
        return self.get_sense_embeddings(slang_ind, fold_name)
        
    def get_sense_embeddings(self, slang_ind, fold_name='default'):
                    
        if self.verbose:
            print("Encoding sense definitions...")
            
        out_dir = Path(self.out_dir) / fold_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Keep encoder identifiers out of the directory structure, including
        # names produced by older encoders with mixed path separators.
        encoder_name = Path(str(self.sense_encoder.name).replace('\\', '/')).name
        filename = "sum_embed_" + encoder_name + ".npz"
            
        sense_embeds = self.sense_encoder.encode_dataset(self.dataset, slang_ind)
        np.savez(out_dir / filename, train=sense_embeds['train'], dev=sense_embeds['dev'], test=sense_embeds['test'], standard=sense_embeds['standard'])
        
        if self.verbose:
            print("Complete!")
            
        return sense_embeds
    
    def get_testtime_embeddings(self, slang_def_sents, fold_name='default', model_path='SBERT_contrastive'):
        
        # Set up directory structure based on model_path
        if model_path == 'SBERT_contrastive' or model_path == 'SBERT_t5' or model_path == 'SBERT-multilingual-MiniLM-L12-v2' or model_path == 'SBERT_LaBSE' or model_path == 'SBERT_mpnet' or model_path == 'SBERT_e5_base' or model_path == 'SBERT_e5_large':
            model_name = self.out_dir + '/' + fold_name + '/SBERT_data/' + model_path
        else:
            raise ValueError(f"Unsupported embed_name: {model_path}")
            
        self.load_sense_encoder(model_name, model_path)
        
        return self.sense_encoder.encode_sentences(slang_def_sents)
        
        
    # def train_contrastive_model(self, slang_ind, params=None, fold_name='default'):
        
    #     if params is None:
    #         params = {'train_batch_size':16, 'num_epochs':4, 'triplet_margin':1, 'outpath':'SBERT_contrastive'}
        
    #     self.prep_contrastive_training(slang_ind, fold_name=fold_name)
        
    #     out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'

    #     triplet_reader = TripletReader(out_dir, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',', has_header=True)
    #     output_path = out_dir+params['outpath']
        
    #     sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        
    #     train_data = SentencesDataset(examples=triplet_reader.get_examples('contrastive_train.csv'), model=sbert_model)
    #     train_dataloader = DataLoader(train_data, shuffle=True, batch_size=params['train_batch_size'])
    #     train_loss = losses.TripletLoss(model=sbert_model, triplet_margin=params['triplet_margin'])

    #     # dev_data = SentencesDataset(examples=triplet_reader.get_examples('contrastive_dev.csv'), model=sbert_model)
    #     # dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=params['train_batch_size'])
    #     # evaluator = TripletEvaluator(dev_dataloader)
        
    #     dev_data = triplet_reader.get_examples('contrastive_dev.csv')
    #     anchors = [ex.texts[0] for ex in dev_data]
    #     positives = [ex.texts[1] for ex in dev_data]
    #     negatives = [ex.texts[2] for ex in dev_data]

    #     evaluator = TripletEvaluator(anchors, positives, negatives, name="dev-eval")

    #     warmup_steps = int(len(train_data)*params['num_epochs']/params['train_batch_size']*0.1) #10% of train data

    #     # Train the model
    #     sbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
    #               evaluator=evaluator,
    #               epochs=params['num_epochs'],
    #               evaluation_steps=len(dev_data),
    #               warmup_steps=warmup_steps,
    #               output_path=output_path)
        
    # only fine-tune the triplet head
    def train_contrastive_model_head(self, slang_ind, params=None, fold_name='default', embed_name='SBERT_contrastive', categorizer='cf_prototype_5', sentence_encoder=None, save_checkpoint=True):
        if params is None:
            params = {'train_batch_size':8, 'num_epochs':4, 'triplet_margin':1, 'outpath':embed_name, 'head_dim':768}

        checkpoint_metric = params.get('checkpoint_metric', 'dev_roc')
        if checkpoint_metric not in ('dev_roc', 'dev_loss'):
            raise ValueError('checkpoint_metric must be dev_roc or dev_loss')
        include_epoch_zero = params.get('include_epoch_zero', False)
        if type(include_epoch_zero) is not bool:
            raise ValueError('include_epoch_zero must be a boolean')
        normalize_triplet_embeddings = params.get('normalize_triplet_embeddings', False)
        if type(normalize_triplet_embeddings) is not bool:
            raise ValueError('normalize_triplet_embeddings must be a boolean')
        resample_negatives = params.get('resample_negatives_each_epoch', False)
        if type(resample_negatives) is not bool:
            raise ValueError('resample_negatives_each_epoch must be a boolean')
        neighbor_weight = params.get('neighbor_loss_weight', 1.0)
        if type(neighbor_weight) not in (int, float) or not np.isfinite(neighbor_weight) or neighbor_weight < 0:
            raise ValueError('neighbor_loss_weight must be a finite nonnegative number')
        weighted_loss = neighbor_weight != 1.0
        if type(params['num_epochs']) is not int or params['num_epochs'] < 1:
            raise ValueError('num_epochs must be a positive integer')
        head_lr = float(params.get('head_lr', 1e-4))
        if not np.isfinite(head_lr) or head_lr <= 0:
            raise ValueError('head_lr must be positive and finite')

        self.prep_contrastive_training(slang_ind, fold_name=fold_name)
        
        # Set up directory structure based on embed_name
        if embed_name == 'SBERT_contrastive' or embed_name == 'SBERT_t5' or embed_name == 'SBERT-multilingual-MiniLM-L12-v2' or embed_name == 'SBERT_LaBSE' or embed_name == 'SBERT_mpnet' or embed_name == 'SBERT_e5_base' or embed_name == 'SBERT_e5_large':
            out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'
        else:
            raise ValueError(f"Unsupported embed_name: {embed_name}")
        output_path = out_dir + params['outpath']

        # Load encoder based on embed_name
        if sentence_encoder is not None:
            se_model = sentence_encoder
        elif embed_name == 'SBERT_contrastive':
            se_model = SentenceTransformer('bert-base-nli-mean-tokens')
        elif embed_name == 'SBERT_t5':
            se_model = SentenceTransformer('sentence-t5-base')
        elif embed_name == 'SBERT-multilingual-MiniLM-L12-v2':
            se_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        elif embed_name == 'SBERT_LaBSE':
            se_model = SentenceTransformer('LaBSE')
        elif embed_name == 'SBERT_mpnet':
            se_model = SentenceTransformer(resolve_sentence_model_path('paraphrase-multilingual-mpnet-base-v2'))
        elif embed_name == 'SBERT_e5_base':  
            se_model = SentenceTransformer(resolve_sentence_model_path('intfloat/multilingual-e5-base'))
        elif embed_name == 'SBERT_e5_large':
            se_model = SentenceTransformer('intfloat/multilingual-e5-large')
        else:
            # Add more cases here for different encoders
            raise ValueError(f"Unsupported embed_name: {embed_name}")
            
        se_model.eval()
        for param in se_model.parameters():
            param.requires_grad = False  # Freeze encoder

        definition_prefix = 'query: ' if embed_name == 'SBERT_e5_base' else ''

        def encode_definitions(sentences, **kwargs):
            if definition_prefix:
                sentences = [definition_prefix + text for text in sentences]
            return se_model.encode(sentences, **kwargs)

        class TripletHead(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
            def forward(self, x):
                return self.linear(x)

        embed_dim = se_model.get_sentence_embedding_dimension()
        head_dim = params.get('head_dim', 768)
        triplet_head = TripletHead(embed_dim, head_dim).cuda() if torch.cuda.is_available() else TripletHead(embed_dim, head_dim)
        initialize_identity_head(triplet_head.linear)

        # Triplet Loss
        triplet_loss_fn = nn.TripletMarginLoss(margin=params['triplet_margin'], p=2,
                                             reduction='none' if weighted_loss else 'mean')

        def compute_triplet_loss(anchor, positive, negative, weights=None, normalizer=1.0):
            if normalize_triplet_embeddings:
                anchor, positive, negative = (F.normalize(x, p=2, dim=-1)
                                              for x in (anchor, positive, negative))
            losses = triplet_loss_fn(anchor, positive, negative)
            if weighted_loss:
                weights = torch.as_tensor(weights, dtype=losses.dtype, device=losses.device)
                # Fixed split-wide denominator avoids cancelling weights in
                # all-neighbor minibatches (or when batch size is one).
                return (losses * weights).mean() / normalizer
            return losses

        # Optimizer (only head)
        optimizer = optim.Adam(triplet_head.parameters(), lr=head_lr)

        # Prepare triplet data
        class TripletDataset(Dataset):
            def __init__(self, anchors, positives, negatives, weights):
                self.anchors = anchors
                self.positives = positives
                self.negatives = negatives
                self.weights = weights
            def __len__(self):
                return len(self.anchors)
            def __getitem__(self, idx):
                return self.anchors[idx], self.positives[idx], self.negatives[idx], self.weights[idx]

        def triplet_collate(batch):
            return list(zip(*batch))
        
        # Load triplets from CSV
        train_df = read_triplet_csv(out_dir+'contrastive_train.csv')
        dev_df = read_triplet_csv(out_dir+'contrastive_dev.csv')
        split_weights = {split: np.ones(len(frame), dtype=np.float32)
                         for split, frame in [('train', train_df), ('dev', dev_df)]}
        if weighted_loss or resample_negatives:
            pools = self.load_preprocessed_data(fold_name=fold_name)
            train_triplets = np.load(out_dir+'triplets.npy').copy()
            if weighted_loss:
                for split, triplets in [('train', train_triplets),
                                        ('dev', np.load(out_dir+'triplets_dev.npy'))]:
                    # Use definition indices, not text equality: distinct words
                    # can share a definition, and old words have multiple senses.
                    gold = [set(row['positive'].tolist()) for row in pools['cp_'+split]]
                    split_weights[split] = np.asarray([
                        1.0 if int(p) in gold[int(a)] else neighbor_weight
                        for a, p, _ in triplets], dtype=np.float32)
                    if len(split_weights[split]) == 0 or split_weights[split].mean() <= 0:
                        raise ValueError(f'{split}: triplet weights must have a positive total')
                    np.save(out_dir+f'triplet_weights_{split}.npy', split_weights[split])
            if resample_negatives:
                conv_texts = [prepare_definition_text(d['def']) for word in self.dataset.vocab
                              for d in self.dataset.conv_data[word].definitions]
        weight_normalizers = {split: float(weights.mean()) for split, weights in split_weights.items()}
        anchors = train_df['anchor'].tolist()
        positives = train_df['positive'].tolist()
        negatives = train_df['negative'].tolist()
        train_dataset = TripletDataset(anchors, positives, negatives, split_weights['train'])
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True, collate_fn=triplet_collate)

        # Create validation dataset and loader
        dev_anchors = dev_df['anchor'].tolist()
        dev_positives = dev_df['positive'].tolist()
        dev_negatives = dev_df['negative'].tolist()
        dev_dataset = TripletDataset(dev_anchors, dev_positives, dev_negatives, split_weights['dev'])
        dev_loader = DataLoader(dev_dataset, batch_size=params['train_batch_size'], shuffle=False, collate_fn=triplet_collate)

        # Move models to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        se_model = se_model.to(device)
        triplet_head = triplet_head.to(device)

        # Training loop (~1/10 tqdm refreshes vs default: update every ~10% of batches)
        _tqdm_miniters = max(1, len(train_loader) // 10)
        _tqdm_miniters_dev = max(1, len(dev_loader) // 10)
        best_score, best_epoch, history = None, None, []
        if checkpoint_metric == 'dev_roc':
            from .model import SlangGenModel
            dev_model = SlangGenModel(self)
            # Frozen encoder outputs are cached once. Test slang is never encoded
            # here; candidate conventional definitions remain available as usual.
            sentences = {split: [prepare_definition_text(self.dataset.slang_data[i].def_sent)
                                 for i in getattr(slang_ind, split)] for split in ('train', 'dev')}
            sentences['standard'] = [prepare_definition_text(d['def']) for word in self.dataset.vocab
                                     for d in self.dataset.conv_data[word].definitions]
            with torch.no_grad():
                base_embeds = {split: encode_definitions(texts, convert_to_tensor=True).to(device)
                               for split, texts in sentences.items()}

        def validate_and_save(epoch, train_loss):
            nonlocal best_score, best_epoch
            triplet_head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc=f'Epoch {epoch} val',
                                  miniters=_tqdm_miniters_dev, mininterval=1.0):
                    anchor_sents, pos_sents, neg_sents, weights = batch
                    embeds = [torch.tensor(encode_definitions(sents, convert_to_numpy=True)).to(device)
                              for sents in (anchor_sents, pos_sents, neg_sents)]
                    batch_loss = compute_triplet_loss(*(triplet_head(x) for x in embeds),
                        weights=weights, normalizer=weight_normalizers['dev']).item()
                    val_loss += batch_loss * (len(weights) if weighted_loss else 1)
                val_loss /= len(dev_dataset) if weighted_loss else len(dev_loader)
                dev_roc = None
                if checkpoint_metric == 'dev_roc':
                    projected = {split: normalize_L2(triplet_head(x).cpu().numpy(), axis=1)
                                 for split, x in base_embeds.items()}
                    dev_roc = dev_model.evaluate_dev_roc(projected, slang_ind, categorizer,
                                                        os.path.join(out_dir, 'dev_selection'))
            score = dev_roc if checkpoint_metric == 'dev_roc' else -val_loss
            if not np.isfinite(score):
                raise ValueError('Nonfinite checkpoint selection metric')
            # A tie retains the earlier eligible epoch.
            improved = best_score is None or score > best_score
            if improved:
                best_score, best_epoch = score, epoch
                if save_checkpoint:
                    torch.save({'se_model': se_model.state_dict(), 'triplet_head': triplet_head.state_dict(),
                                'head_dim': head_dim, 'embed_dim': embed_dim, 'val_loss': val_loss,
                                'dev_roc': dev_roc, 'epoch': epoch, 'checkpoint_metric': checkpoint_metric,
                                'head_lr': head_lr,
                                'definition_prefix': definition_prefix,
                                'resample_negatives_each_epoch': resample_negatives,
                                'neighbor_loss_weight': neighbor_weight,
                                'normalize_triplet_embeddings': normalize_triplet_embeddings}, output_path + '_with_head.pt')
            history.append({'epoch': epoch, 'train_loss': train_loss, 'dev_loss': val_loss, 'dev_roc': dev_roc})
            for row in history:
                row['selected'] = row['epoch'] == best_epoch
            pd.DataFrame(history).to_csv(os.path.join(out_dir, 'epoch_metrics.csv'), index=False)
            print(f'Epoch {epoch}: dev_loss={val_loss:.4f}, dev_roc={dev_roc}, selected_epoch={best_epoch}')

        if checkpoint_metric == 'dev_roc' and include_epoch_zero:
            validate_and_save(0, None)
        for epoch in range(params['num_epochs']):
            if resample_negatives:
                # Epoch 1 uses the original draw. Resample training negatives
                # only from epoch 2; dev triplets stay fixed for comparison.
                if epoch > 0:
                    for row in train_triplets:
                        row[2] = np.random.choice(pools['cp_train'][int(row[0])]['negative'])
                    train_dataset.negatives = [conv_texts[int(n)] for n in train_triplets[:, 2]]
                np.save(out_dir+f'triplets_epoch_{epoch+1:03d}.npy', train_triplets)
            # Training phase
            triplet_head.train()
            total_loss = 0
            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{params['num_epochs']} train",
                miniters=_tqdm_miniters,
                mininterval=1.0,
            ):
                anchor_sents, pos_sents, neg_sents, weights = batch
                if weighted_loss and not any(weights):
                    continue  # No Adam momentum update for a zero-weight batch.
                # Encode sentences (no grad for encoder)
                with torch.no_grad():
                    anchor_emb = torch.tensor(encode_definitions(anchor_sents, convert_to_numpy=True)).to(device)
                    pos_emb = torch.tensor(encode_definitions(pos_sents, convert_to_numpy=True)).to(device)
                    neg_emb = torch.tensor(encode_definitions(neg_sents, convert_to_numpy=True)).to(device)
                # Pass through Triplet Head
                anchor_proj = triplet_head(anchor_emb)
                pos_proj = triplet_head(pos_emb)
                neg_proj = triplet_head(neg_emb)
                # Compute loss
                loss = compute_triplet_loss(anchor_proj, pos_proj, neg_proj,
                    weights=weights, normalizer=weight_normalizers['train'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * (len(weights) if weighted_loss else 1)
            avg_train_loss = total_loss / (len(train_dataset) if weighted_loss else len(train_loader))
            print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")

            validate_and_save(epoch + 1, avg_train_loss)

        print(f'Done. checkpoint_metric={checkpoint_metric}, selected_epoch={best_epoch}, score={best_score}')
    
    def prep_contrastive_training(self, slang_ind, fold_name='default'):
        
        if self.verbose:
            print("Generating triplet data for contrastive training...")
        
        out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'
        create_directory(out_dir)
        
        preproc_data = self.load_preprocessed_data(fold_name=fold_name)
        
        N_train, triplets = self.sample_triplets(preproc_data['cp_train'])
        N_dev, triplets_dev = self.sample_triplets(preproc_data['cp_dev'])
        
        np.save(out_dir+'triplets.npy', triplets)
        np.save(out_dir+'triplets_dev.npy', triplets_dev)
            
        slang_def_sents = []
        for i in range(self.dataset.N_total):
            slang_def_sents.append(prepare_definition_text(self.dataset.slang_data[i].def_sent))

        conv_def_sents = []
        for i in range(self.dataset.V):
            word = self.dataset.vocab[i]
            for d in self.dataset.conv_data[word].definitions:
                conv_def_sents.append(prepare_definition_text(d['def']))
        
        data_train = {'anchor':[slang_def_sents[slang_ind.train[triplets[i][0]]] for i in range(N_train)],\
                      'positive':[conv_def_sents[triplets[i][1]] for i in range(N_train)],\
                      'negative':[conv_def_sents[triplets[i][2]] for i in range(N_train)]}

        data_dev = {'anchor':[slang_def_sents[slang_ind.dev[triplets_dev[i][0]]] for i in range(N_dev)],\
                    'positive':[conv_def_sents[triplets_dev[i][1]] for i in range(N_dev)],\
                    'negative':[conv_def_sents[triplets_dev[i][2]] for i in range(N_dev)]}
        
        df_train = pd.DataFrame(data=data_train)
        df_dev = pd.DataFrame(data=data_dev)
        
        df_train.to_csv(out_dir+'contrastive_train.csv', index=False)
        df_dev.to_csv(out_dir+'contrastive_dev.csv', index=False)
        
        if self.verbose:
            print("Complete!")
        
    def sample_triplets(self, contrast_data):
    
        # Maximum number of positive pairs from the same positive definition
        MAX_PER_POSDEF = 1000
    
        triplets = []

        N_def = contrast_data.shape[0]

        for i in range(N_def):
            anchor = i
            if contrast_data[i]['negative'].shape[0] == 0:
                continue
            pre_pos = -100
            num_d = 0
            
            for positive in np.concatenate([contrast_data[i]['positive'], contrast_data[i]['neighbors']]):
                if positive != pre_pos+1:
                    num_d = MAX_PER_POSDEF
                pre_pos = positive
                if num_d > 0:
                    num_d -= 1

                    negative = np.random.choice(contrast_data[i]['negative'])
                    triplets.append(Triplet(anchor, positive, negative))

        N_triplets = len(triplets)

        if self.verbose:
            print("Sampled %d Triplets" % N_triplets)

        return N_triplets, np.asarray(triplets)
    
    def preprocess_word_dist(self):
        model_name = "paraphrase-multilingual-mpnet-base-v2"
        if self.verbose:
            print(f"Encoding vocab for word_dist with {model_name} ...")

        vocab_texts = [str(w) if w is not None else "" for w in self.dataset.vocab]
        mpnet_encoder = SentenceTransformer(resolve_sentence_model_path(model_name))
        vocab_conv_embeds = np.asarray(
            mpnet_encoder.encode(
                vocab_texts,
                batch_size=128,
                show_progress_bar=self.verbose,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

        return dist.squareform(dist.pdist(vocab_conv_embeds, metric='cosine'))

    # def preprocess_word_dist(self):
        
    #     vocab_conv_embeds = np.zeros((self.dataset.V, self.word_encoder.E))

    #     for i in range(self.dataset.V):
    #         if self.dataset.vocab[i] in self.word_encoder.vocab:
    #             vocab_conv_embeds[i,:] = self.word_encoder.norm_embed(self.dataset.vocab[i])
    #         else:
    #             c_words = self.dataset.vocab[i].split(' ')
    #             count = 0
    #             if len(c_words) > 1:
    #                 embed = np.zeros(self.word_encoder.E)
    #                 for w in c_words:
    #                     if w in self.word_encoder.vocab:
    #                         embed = embed + self.word_encoder.norm_embed(w)
    #                         count += 1
    #                 if count > 0:
    #                     vocab_conv_embeds[i,:] = embed / float(count)

    #             if count == 0:
    #                 vocab_conv_embeds[i,:] = self.word_encoder.norm_embed('unk')

    #     return dist.squareform(dist.pdist(vocab_conv_embeds, metric='cosine'))

    def preprocess_contrastive(self, slang_ind):
        
        Neigh_pivot = int(np.ceil(self.dataset.V/5.0))
        N_neighbor = min(self.MAX_NEIGHBOR, self.dataset.V - Neigh_pivot)

        self.neighbors = np.zeros((self.dataset.V, N_neighbor), dtype=np.int32)
        self.neighbors_close = np.zeros((self.dataset.V, 5), dtype=np.int32)
        for i in range(self.dataset.V):
            self.neighbors[i,:] = np.argsort(self.word_dist[i,:])[max(Neigh_pivot, self.dataset.V-self.MAX_NEIGHBOR):]
            self.neighbors_close[i,:] = np.argsort(self.word_dist[i,:])[1:6]
            
        contrastive_pairs_train = self.compute_contrastive(slang_ind.train)
        contrastive_pairs_dev = self.compute_contrastive(slang_ind.dev)
        
        return contrastive_pairs_train, contrastive_pairs_dev
            
    def compute_contrastive(self, ind):
        
        def get_conv_definds(word_ind):
            return [self.conv_acc[word_ind]+j for j in range(self.conv_lens[word_ind])]
        
        contrast_data = np.empty(ind.shape[0], dtype=object)

        for i in trange(ind.shape[0]):
            word_ind = self.dataset.vocab_ids[ind[i]]
            contrast_data[i] = {}

            positives = [self.conv_acc[word_ind]+j for j in range(self.conv_lens[word_ind])]

            negatives = []
            conv_self = [d['def'] for d in self.dataset.conv_data[self.dataset.vocab[word_ind]].definitions]
            for far_word in self.neighbors[word_ind]:
                conv_defs = [d['def'] for d in self.dataset.conv_data[self.dataset.vocab[far_word]].definitions]
                for j in range(self.conv_lens[far_word]):
                    cand = self.conv_acc[far_word] + j
                    if not is_close_def(self.dataset.slang_data[ind[i]].def_sent, conv_defs[j], threshold=0.2):
                        has_close_cf_def = False
                        for self_def in conv_self:
                            if is_close_def(self_def, conv_defs[j], threshold=0.2):
                                has_close_cf_def = True
                                break
                        if not has_close_cf_def:
                            negatives.append(cand)            
            
            neigh_defs = []
            for close_word in self.neighbors_close[word_ind]:
                neigh_defs.extend([self.conv_acc[close_word]+j for j in range(self.conv_lens[close_word])])

            contrast_data[i]['positive'] = np.asarray(positives)
            contrast_data[i]['negative'] = np.asarray(negatives)
            contrast_data[i]['neighbors'] = np.asarray(neigh_defs)

        return contrast_data

    def train_contrastive_model_whole(self, slang_ind, params=None, fold_name='default', embed_name='SBERT_contrastive'):
        if params is None:
            params = {'train_batch_size':8, 'num_epochs':4, 'triplet_margin':1, 'outpath':embed_name, 'head_dim':768, 'sbert_lr':5e-4, 'head_lr':1e-4}

        self.prep_contrastive_training(slang_ind, fold_name=fold_name)
        
        # Set up directory structure based on embed_name
        if embed_name == 'SBERT_contrastive' or embed_name == 'SBERT_t5' or embed_name == 'SBERT-multilingual-MiniLM-L12-v2' or embed_name == 'SBERT_LaBSE' or embed_name == 'SBERT_mpnet' or embed_name == 'SBERT_e5_base' or embed_name == 'SBERT_e5_large':
            out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'
        else:
            raise ValueError(f"Unsupported embed_name: {embed_name}")
        output_path = out_dir + params['outpath']

        # Load encoder based on embed_name
        if embed_name == 'SBERT_contrastive':
            se_model = SentenceTransformer('bert-base-nli-mean-tokens')
        elif embed_name == 'SBERT_t5':
            se_model = SentenceTransformer('sentence-t5-base')
        elif embed_name == 'SBERT-multilingual-MiniLM-L12-v2':
            se_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        elif embed_name == 'SBERT_LaBSE':
            se_model = SentenceTransformer('LaBSE')
        elif embed_name == 'SBERT_mpnet':
            se_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        elif embed_name == 'SBERT_e5_base':
            se_model = SentenceTransformer('intfloat/multilingual-e5-base')
        elif embed_name == 'SBERT_e5_large':
            se_model = SentenceTransformer('intfloat/multilingual-e5-large')
        else:
            # Add more cases here for different encoders
            raise ValueError(f"Unsupported embed_name: {embed_name}")
            
        # Set SBERT to training mode and enable gradients
        se_model.train()
        for param in se_model.parameters():
            param.requires_grad = True  # Enable gradients for SBERT

        class TripletHead(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
            def forward(self, x):
                return self.linear(x)

        embed_dim = se_model.get_sentence_embedding_dimension()
        head_dim = params.get('head_dim', 768)
        triplet_head = TripletHead(embed_dim, head_dim).cuda() if torch.cuda.is_available() else TripletHead(embed_dim, head_dim)
        initialize_identity_head(triplet_head.linear)

        # Triplet Loss
        triplet_loss_fn = nn.TripletMarginLoss(margin=params['triplet_margin'], p=2)

        # Optimizer 
        sbert_lr = params.get('sbert_lr', 1e-4)
        head_lr = params.get('head_lr', 1e-4)
        optimizer = optim.Adam([
            {'params': se_model.parameters(), 'lr': sbert_lr},
            {'params': triplet_head.parameters(), 'lr': head_lr}
        ])

        # Prepare triplet data
        class TripletDataset(Dataset):
            def __init__(self, anchors, positives, negatives):
                self.anchors = anchors
                self.positives = positives
                self.negatives = negatives
            def __len__(self):
                return len(self.anchors)
            def __getitem__(self, idx):
                return self.anchors[idx], self.positives[idx], self.negatives[idx]

        def triplet_collate(batch):
            return list(zip(*batch)) 
        
        # Load triplets from CSV
        train_df = read_triplet_csv(out_dir+'contrastive_train.csv')
        dev_df = read_triplet_csv(out_dir+'contrastive_dev.csv')
        anchors = train_df['anchor'].tolist()
        positives = train_df['positive'].tolist()
        negatives = train_df['negative'].tolist()
        train_dataset = TripletDataset(anchors, positives, negatives)
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True, collate_fn=triplet_collate)

        # Create validation dataset and loader
        dev_anchors = dev_df['anchor'].tolist()
        dev_positives = dev_df['positive'].tolist()
        dev_negatives = dev_df['negative'].tolist()
        dev_dataset = TripletDataset(dev_anchors, dev_positives, dev_negatives)
        dev_loader = DataLoader(dev_dataset, batch_size=params['train_batch_size'], shuffle=False, collate_fn=triplet_collate)

        # Move models to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        se_model = se_model.to(device)
        triplet_head = triplet_head.to(device)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(params['num_epochs']):
            # Training phase
            se_model.train()
            triplet_head.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']} - Training"):
                anchor_sents, pos_sents, neg_sents = batch
                
                # Encode sentences with gradients (end-to-end training)
                anchor_emb = torch.tensor(se_model.encode(anchor_sents, convert_to_numpy=True)).to(device)
                pos_emb = torch.tensor(se_model.encode(pos_sents, convert_to_numpy=True)).to(device)
                neg_emb = torch.tensor(se_model.encode(neg_sents, convert_to_numpy=True)).to(device)
                
                # Pass through Triplet Head
                anchor_proj = triplet_head(anchor_emb)
                pos_proj = triplet_head(pos_emb)
                neg_proj = triplet_head(neg_emb)
                
                # Compute loss
                loss = triplet_loss_fn(anchor_proj, pos_proj, neg_proj)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")

            # Validation phase
            se_model.eval()
            triplet_head.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']} - Validation"):
                    anchor_sents, pos_sents, neg_sents = batch
                    anchor_emb = torch.tensor(se_model.encode(anchor_sents, convert_to_numpy=True)).to(device)
                    pos_emb = torch.tensor(se_model.encode(pos_sents, convert_to_numpy=True)).to(device)
                    neg_emb = torch.tensor(se_model.encode(neg_sents, convert_to_numpy=True)).to(device)
                    anchor_proj = triplet_head(anchor_emb)
                    pos_proj = triplet_head(pos_emb)
                    neg_proj = triplet_head(neg_emb)
                    loss = triplet_loss_fn(anchor_proj, pos_proj, neg_proj)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(dev_loader)
            print(f"Epoch {epoch+1} average validation loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_dict = {
                    'se_model': se_model.state_dict(),
                    'triplet_head': triplet_head.state_dict(),
                    'head_dim': head_dim,
                    'embed_dim': embed_dim,
                    'val_loss': best_val_loss,
                    'sbert_lr': sbert_lr,
                    'head_lr': head_lr
                }
                torch.save(save_dict, output_path + '_whole_finetuned.pt')
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    def train_multilingual_knowledge_distillation(self, teacher_model_path, sentence_pairs_csv, params=None, fold_name='default', embed_name='SBERT_mpnet'):
        """
        Train a multilingual student model to learn from a teacher model using parallel sentence pairs.
        
        Args:
            teacher_model_path: Path to the trained teacher model (slang-aware)
            sentence_pairs_csv: Path to CSV file with parallel sentence pairs (sentence_s, sentence_t)
            params: Training parameters
            fold_name: Folder name for output
            embed_name: Name of the multilingual student model
        """
        if params is None:
            params = {
                'train_batch_size': 32,
                'num_epochs': 10,
                'learning_rate': 2e-5,
                'head_dim': 768,
                'alpha': 0.5,  # kept for backward-compat (unused when loss=In_Batch_Neg)
                'outpath': embed_name,
                'temperature': 0.07,
                'loss': 'In_Batch_Neg'  # 'In_Batch_Neg' or 'mse'
            }

        # Set up directory structure based on embed_name
        if embed_name == 'SBERT_contrastive' or embed_name == 'SBERT_t5' or embed_name == 'SBERT-multilingual-MiniLM-L12-v2' or embed_name == 'SBERT_LaBSE' or embed_name == 'SBERT_mpnet' or embed_name == 'SBERT_e5_base' or embed_name == 'SBERT_e5_large':
            out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'
        else:
            raise ValueError(f"Unsupported embed_name: {embed_name}")
        
        create_directory(out_dir)
        output_path = out_dir + params['outpath']

        # Load teacher model
        if self.verbose:
            print("Loading teacher model...")
        
        # Determine teacher model type based on path
        teacher_model_name = teacher_model_path.split('/')[-1]
        if 'contrastive' in teacher_model_name:
            teacher_sbert_name = 'bert-base-nli-mean-tokens'
        elif 't5' in teacher_model_name:
            teacher_sbert_name = 'sentence-t5-base'
        elif 'multilingual-MiniLM' in teacher_model_name:
            teacher_sbert_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        elif 'LaBSE' in teacher_model_name:
            teacher_sbert_name = 'LaBSE'
        elif 'mpnet' in teacher_model_name:
            teacher_sbert_name = 'paraphrase-multilingual-mpnet-base-v2'
        elif 'e5_base' in teacher_model_name:
            teacher_sbert_name = 'intfloat/multilingual-e5-base'
        elif 'e5_large' in teacher_model_name:
            teacher_sbert_name = 'intfloat/multilingual-e5-large'
        else:
            raise ValueError(f"Unsupported teacher model: {teacher_model_name}")
        
        teacher_model = SBertWithHeadEncoder(teacher_sbert_name, teacher_model_path)
        teacher_model.sbert_model.eval()
        teacher_model.triplet_head.eval()
        
        if self.verbose:
            print("Teacher model loaded successfully!")

        # Load student model based on embed_name
        if embed_name == 'SBERT_contrastive':
            student_model = SentenceTransformer('bert-base-nli-mean-tokens')
        elif embed_name == 'SBERT_t5':
            student_model = SentenceTransformer('sentence-t5-base')
        elif embed_name == 'SBERT-multilingual-MiniLM-L12-v2':
            student_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        elif embed_name == 'SBERT_LaBSE':
            student_model = SentenceTransformer('LaBSE')
        elif embed_name == 'SBERT_mpnet':
            student_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        elif embed_name == 'SBERT_e5_base':
            student_model = SentenceTransformer('intfloat/multilingual-e5-base')
        elif embed_name == 'SBERT_e5_large':
            student_model = SentenceTransformer('intfloat/multilingual-e5-large')
        else:
            raise ValueError(f"Unsupported embed_name: {embed_name}")

        # Create head for student model
        class StudentHead(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
            def forward(self, x):
                return self.linear(x)

        embed_dim = student_model.get_sentence_embedding_dimension()
        head_dim = params.get('head_dim', 768)
        student_head = StudentHead(embed_dim, head_dim).cuda() if torch.cuda.is_available() else StudentHead(embed_dim, head_dim)

        # Load parallel sentence pairs
        if self.verbose:
            print("Loading parallel sentence pairs...")
        
        pairs_df = pd.read_csv(sentence_pairs_csv)
        pairs_df.dropna(subset=['sentence_s', 'sentence_t'], inplace=True)
        pairs_df[['sentence_s', 'sentence_t']] = pairs_df[['sentence_s', 'sentence_t']].astype(str)
        # Use only the first N samples to speed up experiments
        max_rows = 1500
        if len(pairs_df) > max_rows:
            pairs_df = pairs_df.iloc[:max_rows].reset_index(drop=True)
        
        # Split into train/dev
        train_size = int(0.8 * len(pairs_df))
        train_df = pairs_df.iloc[:train_size]
        dev_df = pairs_df.iloc[train_size:]
        
        if self.verbose:
            print(f"Loaded {len(pairs_df)} sentence pairs ({len(train_df)} train, {len(dev_df)} dev)")

        # Prepare datasets
        class ParallelSentenceDataset(Dataset):
            def __init__(self, source_sents, target_sents):
                self.source_sents = source_sents
                self.target_sents = target_sents
            def __len__(self):
                return len(self.source_sents)
            def __getitem__(self, idx):
                return self.source_sents[idx], self.target_sents[idx]

        def parallel_collate(batch):
            return list(zip(*batch))

        train_dataset = ParallelSentenceDataset(
            train_df['sentence_s'].tolist(), 
            train_df['sentence_t'].tolist()
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['train_batch_size'], 
            shuffle=True, 
            collate_fn=parallel_collate
        )

        dev_dataset = ParallelSentenceDataset(
            dev_df['sentence_s'].tolist(), 
            dev_df['sentence_t'].tolist()
        )
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=params['train_batch_size'], 
            shuffle=False, 
            collate_fn=parallel_collate
        )

        # Move models to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        teacher_model.sbert_model = teacher_model.sbert_model.to(device)
        teacher_model.triplet_head = teacher_model.triplet_head.to(device)
        student_model = student_model.to(device)
        student_head = student_head.to(device)

        # Loss setup
        mse_loss_fn = nn.MSELoss()
        alpha = params.get('alpha', 0.5)
        temperature = params.get('temperature', 0.07)
        _loss_name = str(params.get('loss', 'In_Batch_Neg')).lower()
        use_infonce = (_loss_name in ('in_batch_neg', 'infonce'))

        # Optimizer
        optimizer = optim.Adam([
            {'params': student_model.parameters(), 'lr': params['learning_rate']},
            {'params': student_head.parameters(), 'lr': params['learning_rate']}
        ])

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(params['num_epochs']):
            # Training phase
            student_model.train()
            student_head.train()
            total_loss = 0
            total_mse_loss = 0
            total_distill_loss = 0
            total_infonce_src = 0
            total_infonce_tgt = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']} - Training"):
                source_sents, target_sents = batch
                
                # Get teacher embeddings (no gradients)
                with torch.no_grad():
                    teacher_source_emb = torch.tensor(
                        teacher_model.encode_sentences(source_sents), 
                        dtype=torch.float32
                    ).to(device)
                
                # Get student embeddings (train full model via forward pass)
                src_tokens = student_model.tokenize(list(source_sents))
                src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
                tgt_tokens = student_model.tokenize(list(target_sents))
                tgt_tokens = {k: v.to(device) for k, v in tgt_tokens.items()}
                student_source_emb = student_model(src_tokens)['sentence_embedding']
                student_target_emb = student_model(tgt_tokens)['sentence_embedding']
                
                # Pass through student head
                student_source_head = student_head(student_source_emb)
                student_target_head = student_head(student_target_emb)

                # Normalize embeddings
                student_source_head = torch.nn.functional.normalize(student_source_head, p=2, dim=1)
                student_target_head = torch.nn.functional.normalize(student_target_head, p=2, dim=1)
                teacher_source_emb = torch.nn.functional.normalize(teacher_source_emb, p=2, dim=1)

                if use_infonce:
                    # InfoNCE with in-batch negatives
                    bsz = teacher_source_emb.size(0)
                    labels = torch.arange(bsz, device=device)
                    # student EN vs teacher EN
                    logits_src = (student_source_head @ teacher_source_emb.t()) / temperature
                    loss_src = F.cross_entropy(logits_src, labels)
                    # student ZH vs teacher EN
                    logits_tgt = (student_target_head @ teacher_source_emb.t()) / temperature
                    loss_tgt = F.cross_entropy(logits_tgt, labels)
                    loss = (loss_src + loss_tgt) * 0.5
                else:
                    # Original MSE-based losses
                    mse_loss = mse_loss_fn(student_source_head, teacher_source_emb)
                    distill_loss = mse_loss_fn(student_target_head, teacher_source_emb)
                    loss = alpha * mse_loss + (1 - alpha) * distill_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                if use_infonce:
                    total_infonce_src += float(loss_src.item())
                    total_infonce_tgt += float(loss_tgt.item())
                else:
                    total_mse_loss += mse_loss.item()
                    total_distill_loss += distill_loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            avg_mse_loss = total_mse_loss / len(train_loader)
            avg_distill_loss = total_distill_loss / len(train_loader)
            
            if use_infonce:
                avg_src = total_infonce_src / max(1, len(train_loader))
                avg_tgt = total_infonce_tgt / max(1, len(train_loader))
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} (In_Batch_Neg-src: {avg_src:.4f}, In_Batch_Neg-tgt: {avg_tgt:.4f})")
            else:
                avg_mse_loss = total_mse_loss / max(1, len(train_loader))
                avg_distill_loss = total_distill_loss / max(1, len(train_loader))
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} "
                      f"(MSE: {avg_mse_loss:.4f}, Distill: {avg_distill_loss:.4f})")

            # Validation phase
            student_model.eval()
            student_head.eval()
            val_loss = 0
            val_mse_loss = 0
            val_distill_loss = 0
            val_infonce_src = 0
            val_infonce_tgt = 0
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']} - Validation"):
                    source_sents, target_sents = batch
                    
                    teacher_source_emb = torch.tensor(
                        teacher_model.encode_sentences(source_sents), 
                        dtype=torch.float32
                    ).to(device)
                    
                    src_tokens = student_model.tokenize(list(source_sents))
                    src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
                    tgt_tokens = student_model.tokenize(list(target_sents))
                    tgt_tokens = {k: v.to(device) for k, v in tgt_tokens.items()}
                    student_source_emb = student_model(src_tokens)['sentence_embedding']
                    student_target_emb = student_model(tgt_tokens)['sentence_embedding']
                    
                    student_source_head = student_head(student_source_emb)
                    student_target_head = student_head(student_target_emb)

                    # Normalize embeddings
                    student_source_head = torch.nn.functional.normalize(student_source_head, p=2, dim=1)
                    student_target_head = torch.nn.functional.normalize(student_target_head, p=2, dim=1)
                    teacher_source_emb = torch.nn.functional.normalize(teacher_source_emb, p=2, dim=1)

                    if use_infonce:
                        bsz = teacher_source_emb.size(0)
                        labels = torch.arange(bsz, device=device)
                        logits_src = (student_source_head @ teacher_source_emb.t()) / temperature
                        loss_src = F.cross_entropy(logits_src, labels)
                        logits_tgt = (student_target_head @ teacher_source_emb.t()) / temperature
                        loss_tgt = F.cross_entropy(logits_tgt, labels)
                        loss = (loss_src + loss_tgt) * 0.5
                        val_infonce_src += float(loss_src.item())
                        val_infonce_tgt += float(loss_tgt.item())
                    else:
                        mse_loss = mse_loss_fn(student_source_head, teacher_source_emb)
                        distill_loss = mse_loss_fn(student_target_head, teacher_source_emb)
                        loss = alpha * mse_loss + (1 - alpha) * distill_loss
                        val_mse_loss += mse_loss.item()
                        val_distill_loss += distill_loss.item()

                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(dev_loader)
            if use_infonce:
                avg_val_src = val_infonce_src / max(1, len(dev_loader))
                avg_val_tgt = val_infonce_tgt / max(1, len(dev_loader))
                print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} (In_Batch_Neg-src: {avg_val_src:.4f}, In_Batch_Neg-tgt: {avg_val_tgt:.4f})")
            else:
                avg_val_mse_loss = val_mse_loss / max(1, len(dev_loader))
                avg_val_distill_loss = val_distill_loss / max(1, len(dev_loader))
                print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} "
                      f"(MSE: {avg_val_mse_loss:.4f}, Distill: {avg_val_distill_loss:.4f})")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_dict = {
                    'student_model': student_model.state_dict(),
                    'student_head': student_head.state_dict(),
                    'embed_dim': embed_dim,
                    'head_dim': head_dim,
                    'val_loss': best_val_loss,
                    'alpha': alpha,
                    'learning_rate': params['learning_rate'],
                    'teacher_model_path': teacher_model_path,
                    'temperature': temperature,
                    'loss': ('In_Batch_Neg' if use_infonce else 'mse')
                }
                torch.save(save_dict, output_path + '_multilingual_distilled.pt')
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        print(f"Multilingual student model saved to: {output_path}_multilingual_distilled.pt")
        
        return output_path + '_multilingual_distilled.pt'
