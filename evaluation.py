# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Evaluation"""

from __future__ import print_function
import os
import sys
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import SCAN, xattn_score_t2i, xattn_score_i2t
from collections import OrderedDict
import time
from torch.autograd import Variable
import nltk


try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, target, batch_size):
    """Encode all images and captions loadable by `data_loader`
    """
    cap_embs = None
    cap_lens = None
    max_n_word = len(target[0])
    lengths = [max_n_word] * batch_size
    # compute the embeddings
    cap_emb, cap_len = model.forward_emb(target, lengths, volatile=True)
    #print(img_emb)

        # cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
    cap_embs = np.zeros((1, max_n_word, cap_emb.size(2)))
        # cap_lens = [0] * len(data_loader.dataset)
    cap_lens = cap_len

    # cache embeddings
    cap_embs[0,:max(lengths),:] = cap_emb[0].data.cpu().numpy().copy()

    # cap_lens = cap_lens.data.cpu().numpy().copy()
    cap_lens = np.array(cap_lens).copy()

    return cap_embs, cap_lens


def evalrank(input_string, img_feature, how_many, model_path, data_path=None, split='dev', fold5=False, gpu_num=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    s_t = time.time()
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)
    print("%s seconds taken to load checkpoint" %(time.time() - s_t))
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    # local dir   
    # opt.vocab_path = '/home/ivy/hard2/scan_data/vocab'
    # docker dir
    opt.vocab_path = '/scan/SCAN/data/vocab'
    
    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)
    # print("opt.vocab_size ", opt.vocab_size)

    print("Loading npy file")
    start_time = time.time()
    # local dir
    # img_embs = np.load('/home/ivy/hard2/scan_out/img_embs.npy')
    img_embs = img_feature
    # docker dir
#img_embs = np.load('/scan/SCAN/numpy_data/img_embs.npy')
    print("%s seconds takes to load npy file" %(time.time() - start_time))

    captions = []
    captions.append(str(input_string))
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

    print('Calculating results...')
    start_time = time.time()
    cap_embs, cap_len = encode_data(model, target, opt.batch_size)
    cap_lens = cap_len[0]
    print("%s seconds takes to calculate results" %(time.time() - start_time))
    print("Caption length with start and end index : ", cap_lens)
    print('Images: %d, Captions: %d' %
           (img_embs.shape[0], cap_embs.shape[0]))


    if not fold5:
        img_embs = np.array(img_embs)
        start = time.time()
        if opt.cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        else:
            raise NotImplementedError
        end = time.time()
        print("calculate similarity time:", end-start)

        # top_10 = np.argsort(sims, axis=0)[-10:][::-1].flatten()
        top_n = np.argsort(sims, axis=0)[-(how_many):][::-1].flatten()
        final_result = list(top_n)

        # 5fold cross-validation, only for MSCOCO
    else:
        for i in range(10):
            if i < 9:
                img_embs_shard = img_embs[i * (img_embs.shape[0]//10):(i+1) * (img_embs.shape[0]//10)]
            else:
                img_embs_shard = img_embs[i * (img_embs.shape[0]//10):]
            cap_embs_shard = cap_embs
            cap_lens_shard = cap_lens
            start = time.time()
            if opt.cross_attn == 't2i':
                sims = shard_xattn_t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            elif opt.cross_attn == 'i2t':
                sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            else:
                raise NotImplementedError
            end = time.time()
            print("calculate similarity time:", end-start)

            top_10 = np.argsort(sims, axis=0)[-10:][::-1].flatten()

            print("Top 10 list for iteration #%d : " %(i+1) + str(top_10 + 5000*i))

        #     r, rt0 = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
        #     print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
        #     ri, rti0 = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
        #     print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
        #
        #     if i == 0:
        #         rt, rti = rt0, rti0
        #     ar = (r[0] + r[1] + r[2]) / 3
        #     ari = (ri[0] + ri[1] + ri[2]) / 3
        #     rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        #     print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
        #     results += [list(r) + list(ri) + [ar, ari, rsum]]
        #
        # print("-----------------------------------")
        # print("Mean metrics: ")
        # mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        # print("rsum: %.1f" % (mean_metrics[10] * 6))
        # print("Average i2t Recall: %.1f" % mean_metrics[11])
        # print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
        #       mean_metrics[:5])
        # print("Average t2i Recall: %.1f" % mean_metrics[12])
        # print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
        #       mean_metrics[5:10])

    # torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')
    return final_result

def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p


def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    ##########
    # n_im_shard = (len(images)-1)/shard_size + 1
    # n_cap_shard = (len(captions)-1)/shard_size + 1
    #
    # d = np.zeros((len(images), len(captions)))
    # for i in range(n_im_shard):
    #     im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
    #     for j in range(n_cap_shard):
    #         sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
    #         cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
    #         im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
    #         s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
    #         l = caplens[cap_start:cap_end]
    #         sim = xattn_score_t2i(im, s, l, opt)
    #         d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    # sys.stdout.write('\n')
    # return d


    n_im_shard = (len(images)-1)/shard_size + 1
    print("bbbbbb", captions[0].shape)
    d = np.zeros((len(images), 1))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        sys.stdout.write('\r>> shard_xattn_t2i batch (%d)' %i)
        im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
        s = Variable(torch.from_numpy(captions[0]), volatile=True).cuda()
        l = caplens
        sim = xattn_score_t2i(im, s, l, opt)
        d[im_start:im_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_i2t(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    # r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # medr = np.floor(np.median(ranks)) + 1
    # meanr = ranks.mean() + 1
    # if return_ranks:
    #     return (r1, r5, r10, medr, meanr), (ranks, top1)
    # else:
    #     return (r1, r5, r10, medr, meanr)
    return top1

def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    # r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # medr = np.floor(np.median(ranks)) + 1
    # meanr = ranks.mean() + 1
    # if return_ranks:
    #     return (r1, r5, r10, medr, meanr), (ranks, top1)
    # else:
    #     return (r1, r5, r10, medr, meanr)
    return top1
