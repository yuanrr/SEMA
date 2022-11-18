from __future__ import print_function

from torch.utils import model_zoo
from torchvision.models.vgg import model_urls

from DAMSM import RNN_ENCODER
import pickle
import dateutil.tz

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import Dataset

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

# from DAMSM import RNN_ENCODER
import torch.nn.functional as F
import os
import sys
import random
import pprint
import datetime
import dateutil.tz
import argparse

from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
from truncation import truncated_noise_sample
from model import NetG

# from model_real import NetG,NetD

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 200
import numpy as np
import multiprocessing

multiprocessing.set_start_method('spawn', True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/coco.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='..')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--trunc', type=bool, default=True)
    parser.add_argument('--truncation', type=float, default=0.8, help='node rank for distributed training')
    args = parser.parse_args()
    return args


def mean_latent(n_latent, style_dim, device):
    latent_in = torch.randn(
        n_latent, style_dim, device=device
    )
    latent = latent_in.mean(0, keepdim=True)
    return latent

def truncate(truncation, truncation_latent, noise):
    if truncation < 1:
        truncated_noise = truncation_latent + truncation * (noise - truncation_latent)
    return truncated_noise


def sampling(text_encoder, netG, dataloader, device):
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    netG.load_state_dict(torch.load('netG.pth'))
    netG.eval()
    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            imags, captions, cap_lens, class_ids, keys, bert = prepare_data(data)
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            # sent_emb = 0.5 * sent_emb.data + 0.5 * bert.data
            sent_emb = sent_emb.data  # the released model use LSTM encoder only
            with torch.no_grad():
                if args.trunc == True:
                    truncation = args.truncation
                    noise = truncated_noise_sample(batch_size=batch_size, truncation=truncation)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                    fake_imgs = netG(noise, sent_emb)
                else:
                    noise = torch.randn(batch_size, 100)
                    noise = noise.to(device)
                    fake_imgs = netG(noise, sent_emb)
            for j in range(batch_size):
                s = '%s/single24110/%s'
                s_tmp = s % (save_dir, keys[j])  # 改这里
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d.png' % (s_tmp, i)
                im.save(fullpath)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')

    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = NetG(cfg.TRAIN.NF, 100).to(device)

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    state_epoch = 0

    count = sampling(text_encoder, netG, dataloader, device)  # generate images for the whole valid dataset
    print('state_epoch:  %d' % (state_epoch))




