import pandas as pd
import numpy as np
import os
import argparse
import json
import torch
from torch import nn
import time
from tqdm import tqdm
import datetime
from gesture_dataset import GestureDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import schedulefree

from model import MyEncoder, MyDecoder, EncodecModel
from qt import ResidualVectorQuantizer, DummyQuantizer

from transformers import get_constant_schedule_with_warmup

def save_checkpoint(model, epoch, rank):
    if rank == 0:
        torch.save(model.module.state_dict(), f"/local2/abzaliev/saved_models/hard_and_yes_ctc_{str(epoch)}.pt")

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def run_epoch(loader, model, optimizer, adversary, balancer, scheduler, epoch, device_id, is_train=True):
    loader.sampler.set_epoch(epoch)
    start = time.time()
    torch.set_grad_enabled(is_train)

    # this if-else below is requirememnt from shedulefree tokenizer, see their docs
    if is_train:
        model.train()
        optimizer.train()
    else:
        optimizer.eval()
        model.train()
        optimizer.eval()
        with torch.no_grad():
            for batch in itertools.islice(loader, 50):
                model(batch)
        model.eval()

    running_loss = 0
    running_disc_loss = 0
    # I experimented with other losses as well the difference is minor
    mse = torch.nn.SmoothL1Loss(reduction='none') # torch.nn.MSELoss(reduction='none') #  # torch.nn.HuberLoss(reduction='none', delta=0.1)
    batch_size = loader.batch_size
    quantizer_loss = 0
    reconstruct_loss = 0
    disc_loss = 0
    disc_counter = 0
    inv_weight = 16.0 # basically just lambda how 
    runnin_ctc_loss = 0
    ctc_loss = nn.CTCLoss(blank=59, zero_infinity=True)


    for ix, batch in (pbar := tqdm(enumerate(loader), total=len(loader))):

        bs = batch['input'].shape[0]
        del (batch['phrase'])
        local_step = ((ix + 1) * batch_size)
        mask = batch['input_mask'].long()
        batch = batch_to_device(batch, device=device_id)
        y = batch['input'].clone() # self.model.encoder.feature_extractor(x['input'], x['input_mask'].long()).transpose(1,2)

        # this is for CTC loss
        targets = batch['token_ids']
        target_mask = batch['attention_mask']
        target_lengths = target_mask.sum(axis=-1).long()
        input_lengths = mask.sum(axis=-1)

        y_mask = mask.clone()
        qres, log_probs = model(batch)
        y_pred = qres.x


        loss = mse(y, y_pred)
        # masking for padding
        loss = (loss.cpu() * mask.unsqueeze(-1).unsqueeze(-1)).sum() 
        non_zero_elements = mask.sum()
        loss = ((loss / non_zero_elements))
        reconstruct_loss += loss.detach()

        ctc = ctc_loss(log_probs, targets, input_lengths, target_lengths).cpu()
        runnin_ctc_loss += ctc.detach()
        loss += ctc

        # penalty from quantization
        quantizer_loss += qres.penalty.cpu().detach()
        loss += ((qres.penalty.cpu())/inv_weight) 
        running_loss += (float(loss.detach()))

        # gradient
        # I think it is related to torchrun
        if is_train:
            loss.backward()
            tb_writer.add_scalar('grad_norm', torch.stack(
                [p.grad.detach().norm() for p in model.module.parameters() if p.grad is not None]).norm().item(), epoch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(epoch + ix / len(loader.dataset))

        status = '{} {}'.format('Train' if is_train else 'Valid', epoch)  # : {:<6}/ {} , local_step, len(pbar)
        status += ' l: {:.4f} avg_l: {:.4f} lr {}'.format(
            loss.item(),  # print batch loss and avg loss
            running_loss / ((ix + 1.0)),  # print batch loss and avg loss
            str(optimizer.param_groups[0]['lr'])[:7])

        pbar.set_description(status)
        disc_counter += 1


    avg_loss = running_loss / ((ix + 1.0))
    # avg_disc_loss = running_disc_loss / ((ix + 1.0))
    tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)

    if is_train:
        tb_writer.add_scalar('loss/train', avg_loss, epoch)
        tb_writer.add_scalar('quantizer_loss/train', quantizer_loss / ((ix + 1.0)), epoch)
        tb_writer.add_scalar('reconstruct_loss/train', reconstruct_loss / ((ix + 1.0)), epoch)
        tb_writer.add_scalar('disc_loss/train', disc_loss / ((ix + 1.0)), epoch)
        tb_writer.add_scalar('ctc_loss/train', runnin_ctc_loss / ((ix + 1.0)), epoch)
        # tb_writer.add_scalar('loss/disc', avg_disc_loss, epoch)
    else:
        tb_writer.add_scalar('quantizer_loss/val', quantizer_loss / ((ix + 1.0)), epoch)
        tb_writer.add_scalar('loss/val', avg_loss, epoch)
        tb_writer.add_scalar('disc_loss/val', disc_loss / ((ix + 1.0)), epoch)
        tb_writer.add_scalar('reconstruct_loss/val', reconstruct_loss / ((ix + 1.0)), epoch)
        tb_writer.add_scalar('ctc_loss/val', runnin_ctc_loss / ((ix + 1.0)), epoch)
    return avg_loss

def run(args):
    init_process_group(backend='nccl')
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    tb_writer = args.tb_writer

    batch_size = args.batch_size // args.num_gpus

    train_ds = GestureDataset(args.train_df, cfg=args.kaggle_gesture_cfg_train,  mode="train")
    val_ds = GestureDataset(args.val_df, cfg=args.kaggle_gesture_cfg_val,  mode="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=16, collate_fn=None,
                              sampler=DistributedSampler(train_ds, shuffle=True))
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=8, collate_fn=None, shuffle=False,
                            sampler=DistributedSampler(val_ds, shuffle=False))

    encoder = MyEncoder(channels=42, dimension=128)
    decoder = MyDecoder(channels=42, dimension=128)
    quantizer = ResidualVectorQuantizer(dimension=encoder.dimension, q_dropout=True, bins=1024, n_q=4)

    model = EncodecModel(encoder, decoder, quantizer, frame_rate=10, sample_rate=10, channels=42)
    model.to(device_id)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[device_id]) 

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=50) # CosineAnnealingLR(optimizer, T_max=300)
    balancer = None
    # balancer = Balancer({'adv': 1, 'rec': 1})

    adv_loss = None

    train_losses = list()
    val_losses = list()
    prev_loss = 100000

    for epoch in range(0, 200):
        train_loss = run_epoch(train_loader, model, optimizer, adv_loss, balancer, scheduler, epoch, device_id, is_train=True)
        train_losses.append(train_loss)
        val_loss = run_epoch(val_loader, model, optimizer, adv_loss, balancer, scheduler, epoch, device_id, is_train=False)
        val_losses.append(val_loss)

        if val_loss < prev_loss:
            save_checkpoint(model, epoch, rank)
            prev_loss = val_loss

if __name__ == "__main__":
    tb_writer = SummaryWriter(log_dir='./new_tb_runs/' + 'back_to_200' + str(datetime.datetime.now().strftime('%d %B %H:%M')))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    # dataset
    sample_rate = 10  # cfg.sample_rate
    channels = 42  # cfg.channelsmse_loss
    seed = 42
    max_sample_rate = 15
    max_channels = 42

    # load dfs
    datamount_path = "/local2/abzaliev/sign_lang/kaggle-asl-fingerspelling-1st-place-solution/datamount"
    df = pd.read_csv(os.path.join(datamount_path, "train_folded_real_lens.csv"))

    # df['is_sup'] = 0
    train_df = df[(df["fold"] != 3) & (df["fold"] != 2)].copy()# .head(2000)
    val_df = df[df["fold"] == 3].copy()# .head(2000)

    # this is the same for both train and validation
    with open(os.path.join(datamount_path, 'character_to_prediction_index.json'), "r") as f:
        char_to_num = json.load(f)

    # rev_character_map = {j: i for i, j in char_to_num.items()}
    with open('/local2/abzaliev/sign_lang/kaggle-asl-fingerspelling-1st-place-solution/datamount/character_to_prediction_index.json',
            "r") as f:
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

    kaggle_gesture_cfg_train = {
        'min_seq_len': 15,
        'data_folder': '/local2/abzaliev/sign_lang/train_landmarks_npy_even_less/',
        'symmetry_fp': os.path.join(datamount_path, 'symmetry.csv'),
        'max_len': 196,
        'flip_aug': 0.25,
        'outer_cutmix_aug': 0.0,
        'max_phrase': 31 + 2,
        'pad_token': 'P',
        'start_token': 'S',
        'end_token': 'E',
        'tokenizer': [char_to_num, num_to_char, chars]
    }

    kaggle_gesture_cfg_val = {
        'min_seq_len': 15,
        'data_folder': '/local2/abzaliev/sign_lang/train_landmarks_npy_even_less/',
        'symmetry_fp': os.path.join(datamount_path, 'symmetry.csv'),
        'max_len': 196,
        'flip_aug': 0.25,
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
    args.batch_size = int(22)
    args.train_df = train_df
    args.val_df = val_df
    args.batch_size = int(args.batch_size * args.num_gpus)
    args.kaggle_gesture_cfg_train = kaggle_gesture_cfg_train
    args.kaggle_gesture_cfg_val = kaggle_gesture_cfg_val
    args.tb_writer = tb_writer
    run(args)