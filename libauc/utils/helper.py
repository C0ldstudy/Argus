import torch 
import numpy as np
import datetime
import os
import sys
import time
import shutil
from tqdm import tqdm, trange
from ..metrics import ndcg_at_k, map_at_k

def batch_to_gpu(batch, device='cuda'):
    for c in batch:
        if type(batch[c]) is torch.Tensor:
            batch[c] = batch[c].to(device)
    return batch

def adjust_lr(learning_rate, lr_schedule, optimizer, epoch):
    lr = learning_rate
    for milestone in eval(lr_schedule):
        lr *= 0.25 if epoch >= milestone else 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate_method(predictions, ratings, topk, metrics):
    """
    :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
    :param ratings: (# of users, # of pos items)
    :param topk: top-K value list
    :param metrics: metric string list
    :return: a result dict, the keys are metric@topk
    """
    evaluations = dict()

    num_of_users, num_pos_items = ratings.shape
    sorted_ratings = -np.sort(-ratings)            # descending order !!
    discounters = np.tile([np.log2(i+1) for i in range(1, 1+num_pos_items)], (num_of_users, 1))
    normalizer_mat = (np.exp2(sorted_ratings) - 1) / discounters

    sort_idx = (-predictions).argsort(axis=1)    # index of sorted predictions (max->min)
    gt_rank = np.array([np.argwhere(sort_idx == i)[:, 1]+1 for i in range(num_pos_items)]).T  # rank of the ground-truth (start from 1)
    for k in topk:
        hit = (gt_rank <= k)
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'NDCG':
                evaluations[key] = ndcg_at_k(ratings, normalizer_mat, hit, gt_rank, k)
            elif metric == 'MAP':
                evaluations[key] = map_at_k(hit, gt_rank)
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    return evaluations

def evaluate(model, data_set, topks, metrics, eval_batch_size=250, num_pos=10):
    """
    The returned prediction is a 2D-array, each row corresponds to all the candidates,
    and the ground-truth item poses the first.
    Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
             predictions like: [[1,3,4], [2,5,6]]
    """
    EVAL_BATCH_SIZE = eval_batch_size
    NUM_POS = num_pos
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = list()
    ratings = list()
    for idx in trange(0, len(data_set), EVAL_BATCH_SIZE):
        batch = data_set.get_batch(idx, EVAL_BATCH_SIZE)
        prediction = model(batch_to_gpu(batch, DEVICE))['prediction']
        predictions.extend(prediction.cpu().data.numpy())
        ratings.extend(batch['rating'].cpu().data.numpy())

    predictions = np.array(predictions)                                 # [# of users, # of items]
    ratings = np.array(ratings)[:, :NUM_POS]                            # [# of users, # of pos items]

    return evaluate_method(predictions, ratings, topks, metrics)

def format_metric(result_dict):
    assert type(result_dict) == dict
    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys()])
    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = '{}@{}'.format(metric, topk)
            m = result_dict[name]
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('{}:{:<.4f}'.format(name, m))
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
