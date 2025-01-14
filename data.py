# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import time

##########
class PrecompDataset(data.Dataset):
# class PrecompDataset(data):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        # self.captions = []
        # with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
        #     for line in f:
        #         self.captions.append(line.strip())

        ##########
        # Captions
        self.captions = []
        self.captions.append(raw_input("Text Query : "))
        # self.captions = 5000 * self.captions

        # Image features ( self.images.shape = (5000,36,2048), self.length = 5000 )
        start_time = time.time()
        # self.images = np.load(loc+'%s_ims.npy' % data_split)
        # self.images = np.load(loc+'testall_ims.npy')
        self.images = np.load('./out/img_embs.npy')
        print("%s seconds taken to load npy data" %(time.time()-start_time))
        # print(".npy file shape : ", self.images.shape)    .npy file shape = (# of images, 36, 2048)
        # self.length = len(self.captions)
        self.length = self.images.shape[0]

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        ##########
        #self.im_div = 1


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        # images = torch.Tensor(self.images)
        #print('image', image.numpy().shape)    (36, 2048)
        #print('index', index)      0, 1, 2, 3, ...

        ##########
        # caption = self.captions[index]
        caption = self.captions

        #print('caption',caption)      One line of captions
        vocab = self.vocab

        ##########
        #caption = self.captions


        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        #print('start', vocab('<start>'))   '1'
        caption.extend([vocab(token) for token in tokens])
        #print('token', [vocab(token) for token in tokens])     Match vocab idx into each caption
        caption.append(vocab('<end>'))
        #print('end', vocab('<end>'))   '2'
        #print('caption size', np.array(caption).shape)     = cap_lens
        #print('caption', caption)      = [1, x, x, x, x, x, ..., x, 2]
        target = torch.Tensor(caption)
        return image, target, index, img_id
        # return images, target

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, image_ids = zip(*data)
    # images, captions = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids
    # return images, targets


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
