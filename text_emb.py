import os
import base64
import csv
import sys
import zlib
import json
import argparse
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch
import time
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from torch.autograd import Variable
from vocab import Vocabulary, deserialize_vocab
import nltk
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


def EncoderImage(img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        # print('cap_emb : ', cap_emb.shape, 'cap_len : ', cap_len.shape)
        # cap_emb.shape : (128, 8, 2048) cap_len.shape : (128,)
        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        # cap_emb.shape : (8, 8, 1024)
        return cap_emb, cap_len

class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)

        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [None, self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.txt_enc.load_state_dict(state_dict[1])

    def forward_emb(self, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            captions = captions.cuda()

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return cap_emb, cap_lens


    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        cap_emb, cap_lens = self.forward_emb(captions, lengths)

def encode_data(model, target):
    """Encode all images and captions loadable by `data_loader`
    """
    cap_embs = None
    cap_lens = None
    max_n_word = len(target[0])
    lengths = [max_n_word] * opt.batch_size
    # compute the embeddings
    cap_emb, cap_len = model.forward_emb(target, lengths, volatile=True)
    #print(img_emb)

        # cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
    cap_embs = np.zeros((1, max_n_word, cap_emb.size(2)))
        # cap_lens = [0] * len(data_loader.dataset)
    cap_lens = cap_len

    # cache embeddings
    cap_embs[0,:max(lengths),:] = cap_emb[0].data.cpu().numpy().copy()

    cap_lens = cap_lens.data.cpu().numpy().copy()

    return cap_embs, cap_lens




if __name__ == '__main__':

    model_path = 'runs/coco_scan/log/model_best.pth.tar'
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    model = SCAN(opt)
    model.load_state_dict(checkpoint['model'])

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    captions = []
    captions.append("A woman is playing a guitar .")
    tokens = nltk.tokenize.word_tokenize(
        str(captions).lower().decode('utf-8'))
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = []
    for batch in range(opt.batch_size):
        target.append(caption)
    target = torch.Tensor(target).long()
    print('target', target)

    # lengths = [len(target)]
    # targets = torch.zeros(1, max(lengths)).long()
    # for i, cap in enumerate(target):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]
    # capt = targets

    cap_embs, cap_lens = encode_data(model, target)

    print('cap_embs\n', cap_embs)
    print('Final numpy shape : ', cap_embs.shape)

