# this script discretizes the whole dataset
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import pandas as pd
import numpy as np

import argparse
import json
import torch
import time
from tqdm import tqdm
from gesture_dataset import GestureDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# import viz_utils
from model import MyEncoder, MyDecoder, EncodecModel
from qt import ResidualVectorQuantizer, DummyQuantizer
# from gpt2_more_inputs import GPT2LMHeadModel
from collections import defaultdict
import itertools

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

sample_rate = 10  # cfg.sample_rate
channels = 130  # cfg.channels
seed = 42
max_sample_rate = 15
max_channels = 130

# load dfs
datamount_path = ".../kaggle-asl-fingerspelling-1st-place-solution/datamount"
df = pd.read_csv(os.path.join(datamount_path, "train_folded_real_lens.csv"))
df['is_sup'] = 0

# take fold 2 as a validation data
train_df = df[(df["fold"] != 3) & (df["fold"] != 2)].copy()
val_df = df[df["fold"] == 2].copy()

# this is the same for both train and validation
with open(os.path.join(datamount_path, 'character_to_prediction_index.json'), "r") as f:
    char_to_num = json.load(f)

rev_character_map = {j: i for i, j in char_to_num.items()}
n = len(char_to_num)
pad_token = 'P'
start_token = 'S'
end_token = 'E'
char_to_num[pad_token] = n
char_to_num[start_token] = n + 1
char_to_num[end_token] = n + 2
num_to_char = {j: i for i, j in char_to_num.items()}
chars = np.array([num_to_char[i] for i in range(len(num_to_char))])

kaggle_gesture_cfg_val = {
    'min_seq_len': 15,
    'data_folder': '.../train_landmarks_npy_even_less/',
    'symmetry_fp': os.path.join(datamount_path, 'symmetry.csv'),
    'max_len': 384,
    'flip_aug': 0.0,
    'outer_cutmix_aug': 0.0,
    'max_phrase': 31 + 2,
    'pad_token': 'P',
    'start_token': 'S',
    'end_token': 'E',
    'tokenizer': [char_to_num, num_to_char, chars]
}

parser = argparse.ArgumentParser()

args = parser.parse_args()
args.num_gpus = torch.cuda.device_count()  
args.batch_size = int(36)
args.train_df = train_df
args.val_df = val_df
args.batch_size = int(args.batch_size * args.num_gpus)
args.kaggle_gesture_cfg_val = kaggle_gesture_cfg_val
device_id = 'cuda:0'

batch_size = 32
train_ds = GestureDataset(args.train_df, cfg=args.kaggle_gesture_cfg_val, mode='test')
val_ds = GestureDataset(args.val_df, cfg=args.kaggle_gesture_cfg_val, mode='test')

train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, collate_fn=None, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=None, shuffle=False)

X_CHANNELS = 42
encoder = MyEncoder(channels=X_CHANNELS, dimension=128)
decoder = MyDecoder(channels=X_CHANNELS, dimension=128)
quantizer = ResidualVectorQuantizer(dimension=encoder.dimension, bins=1024, n_q=4)

model = EncodecModel(encoder, decoder, quantizer, frame_rate=10, sample_rate=10, channels=X_CHANNELS)
model.to(device_id)

model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model.load_state_dict(torch.load('trained_model_from_step2.pt'), strict=False)  # map_location={'cuda:0': 'cuda:1'}
model.train()
with torch.no_grad():
    for batch in itertools.islice(val_loader, 50):
        del (batch['phrase'])
        batch = batch_to_device(batch, device_id)
        model(batch)
model.eval()

print(f"Number of params: {sum(p.numel() for p in model.parameters())}")

generate_codes = True
generate_embeddings = False

if generate_codes:

    # save discrete codes
    model.eval()
    all_discrete_states = list()
    all_gts = list()
    all_preds = list()
    all_phrases = list()

    with torch.no_grad():
        for ix, batch in (pbar := tqdm(enumerate(val_loader), total=len(val_loader))):

            all_phrases.append(batch['phrase'])
            del (batch['phrase'])
            batch = batch_to_device(batch, device=device_id)

            # THIS SLICES THE BATCH INTO EQUAL SLICES oF LENGTH 64
            ########################################################
            chunked = batch['input'].unfold(1, 64, 64)
            chunked_mask = batch['input_mask'].unfold(1, 64, 64)

            chunked_list = [chunked[:,i].permute(0, 3, 1, 2) for i in range(6)]
            chunked_mask_list = [chunked_mask[:,i] for i in range(6)]
            chunked_results = []
            for _inp, _mask in zip(chunked_list, chunked_mask_list):
                new_batch = {'input': _inp, 'input_mask': _mask}
                discrete_pred_nano = model.encode_to_discrete(new_batch).detach().cpu()
                chunked_results.append(discrete_pred_nano)

            # I think it worked
            glued_results = torch.cat(chunked_results, dim=2)
            batch['input'] = batch['input'] # [:,:60]
            batch['input_mask'] = batch['input_mask'] # [:,:60]
            discrete_pred = model.encode_to_discrete(batch).detach()
            emb = model.quantizer.decode(discrete_pred).cpu()
            discrete_pred = discrete_pred.cpu()
            qres, feats_predicted = model(batch)
            y_pred = qres.x

            for iix in range(batch['input'].shape[0]):
                gt = batch['input'][iix].cpu()
                predicted = y_pred[iix].squeeze().cpu()
                mask = batch['input_mask'][iix].bool().cpu()
                
                discrete_codes = discrete_pred[iix][:, mask]
                to_save_gt = gt[mask].cpu()
                to_save_pred = predicted[mask]

                all_discrete_states.append(discrete_codes)
                all_preds.append(to_save_pred)
                all_gts.append(to_save_gt)

    validation_df = pd.DataFrame({'discretes': all_discrete_states, 'preds': all_preds, 'gts': all_gts, 'phrases': [j for i in all_phrases for j in i]})
    validation_df.to_pickle('discrete_dataset.pkl')