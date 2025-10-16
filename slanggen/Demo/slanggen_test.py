import numpy as np
import torch
import nltk
nltk.download('stopwords')

from slanggen.util import *
from slanggen.dataloader import WN_Dataset, Urban_Dataset
from slanggen.encoder import FTEncoder, FTCachedEncoder
from slanggen.contrastive import SlangGenTrainer
from slanggen.model import SlangGenModel

torch.cuda.set_device(0)

# Load conventional definition data using the builtin dataloaders. 
# The .npy file loaded below contains a pre-processed version of WordNet definition sentences for all words that appear in both WordNet and UD.

wn_data = WN_Dataset('wordnet_urban.npy')
print(wn_data)

# Load slang definition data. The .npy file loaded below is a pre-processed version of the data released in this repository.

urban_wn_data = Urban_Dataset('Urban_data.npy', wn_data)
print(urban_wn_data)

# If you wish to use your own dataset, please create a dataloader object inheriting either ConvDataset or SlangDataset abstract classes found in dataloader.py and following the example data specifications in dataloader.WN_Dataset and dataloader.Urban_Dataset.
# Now let's create a directory to store our results and load in some pre-generated data indices for train-test split:

create_directory('Results')
out_dir='Results/'

dataset = urban_wn_data
slang_inds = DataIndex(np.load('train_ind.npy'), np.load('dev_ind.npy'), np.load('test_ind.npy'))

ft_encoder = FTCachedEncoder('ft_embed_cache_Urban.pickle')
trainer = SlangGenTrainer(dataset, word_encoder=ft_encoder, out_dir=out_dir, verbose=True)

# model = SlangGenModel(trainer, data_dir=out_dir)
model = SlangGenModel(trainer, data_dir=out_dir, embed_name='SBERT_contrastive')

params = {'embed_name':'SBERT_contrastive', 'out_name':'predictions', 'model':'cf_prototype_5', 'prior':None, 'prior_name':'uniform', 'contr_params':None}

model.train_contrastive(slang_inds, fold_name='urban_wn', params=params)

model.train_categorization(slang_inds, fold_name='urban_wn', params=params)

results = model.get_results(fold_name='urban_wn', mode='train', params=params)

N_train_dev = dataset.N_total - slang_inds.test.shape[0]
train_rankings = get_rankings(results, np.arange(N_train_dev), dataset.vocab_ids[np.concatenate(slang_inds)])
np.mean(get_roc(train_rankings, dataset.V))

model.predict_testset(slang_inds, fold_name='urban_wn', params=params)

results = model.get_results(fold_name='urban_wn', mode='test', params=params)

N_train_dev = dataset.N_total - slang_inds.test.shape[0]
test_rankings = get_rankings(results, np.arange(N_train_dev, dataset.N_total), dataset.vocab_ids[np.concatenate(slang_inds)])
np.mean(get_roc(test_rankings, dataset.V))

def collect_slang_sents(dataset, ind):
    sentences = []
    for i in ind:
        sentences.append(' '.join(simple_preprocess(dataset.slang_data[i].def_sent)))
    return sentences

test_sents = collect_slang_sents(dataset, slang_inds.test)
test_labels = dataset.vocab_ids[slang_inds.test]

model.predict_from_definitions(test_sents, test_labels, fold_name='urban_wn', mode='testdef', params=params)

results = model.get_results(fold_name='urban_wn', mode='testdef', params=params)

N_train_dev = dataset.N_total - slang_inds.test.shape[0]
test_rankings = get_rankings(results, np.arange(N_train_dev, dataset.N_total), dataset.vocab_ids[np.concatenate(slang_inds)])
np.mean(get_roc(test_rankings, dataset.V))


