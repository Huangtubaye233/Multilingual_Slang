import numpy as np
import torch
import nltk
nltk.download('stopwords')

from slanggen.util import *
from slanggen.dataloader import WN_Dataset, Dummy_ZH_Dataset, Urban_Dataset, OSD_Dataset, ZH_Dataset
from slanggen.encoder import FTEncoder, FTCachedEncoder
from slanggen.contrastive import SlangGenTrainer
from slanggen.model import SlangGenModel

torch.cuda.set_device(0)
# wn_data = WN_Dataset('mix_conv_data.npy')
# dataset = OSD_Dataset('mix_data.npy', wn_data)

ru_data = WN_Dataset('RU_ru_conv_data.npy')
dataset = ZH_Dataset('RU_ru_slang_data.npy', ru_data)

N = len(dataset.slang_data) 

print("N from dataset:", N)

np.random.seed(42)
perm = np.random.permutation(N)

N_train = int(0.8 * N)
N_dev = int(0.05 * N)
N_test = N - N_train - N_dev

train_ind = perm[:N_train]
dev_ind = perm[N_train:N_train + N_dev]
test_ind = perm[N_train + N_dev:]

np.save('train_ind_ru_ru.npy', train_ind)
np.save('dev_ind_ru_ru.npy', dev_ind)
np.save('test_ind_ru_ru.npy', test_ind)

# slang_inds = DataIndex(train_ind, dev_ind, test_ind)

print("✅ Split complete:")
print(f"Train: {len(train_ind)}, Dev: {len(dev_ind)}, Test: {len(test_ind)}")
