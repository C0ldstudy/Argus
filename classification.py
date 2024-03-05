import math
import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as auc_score, f1_score, average_precision_score as ap_score, precision_recall_curve, confusion_matrix
from libauc.optimizers import SOAP
import torch
from torch.optim import Adam, Adadelta
from loaders.tdata import TData
from loaders.load_optc import load_optc_dist
from models.argus import Argus, DetectorEncoder
from utils import get_score, get_optimal_cutoff

TMP_FILE = 'tmp.dat'

def classification(args, rnn_args, worker_args, OUTPATH, device):
    if args.val_times is None:
        val = max((args.tr_end - args.tr_start) // 20, args.delta*2)
        args.val_start = args.tr_end-val
        args.val_end = args.tr_end
        args.tr_end = args.val_start
    else:
        args.val_start = args.val_times[0]
        args.val_end = args.val_times[1]

    times = {
        'tr_start': args.tr_start,
        'tr_end': args.tr_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'te_times': args.te_times,
        'delta': args.delta
    }
    global LOAD_FN
    LOAD_FN = args.loader

    # Evaluating a pre-trained model, so no need to train
    if args.load:
        kwargs = {
            'start': None,
            'end': None,
            'use_flows': args.flows,
            'device': device
        }
        rrefs = args.encoder(LOAD_FN, kwargs, *worker_args)
        rnn = args.rnn(*rnn_args)
        model = Argus(rnn, rrefs, device)

        states = pickle.load(open('./Exps/model_save_'+args.dataset+'.pkl', 'rb'))
        model.load_states(*states['states'])
        h0 = states['h0']
        tpe = 0
        tr_time = 0

    # Building and training a fresh model
    else:
        kwargs = {
                'start': times['tr_start'],
                'end': times['tr_end'],
                'delta': times['delta'],
                'is_test': False,
                'use_flows': args.flows,
                'device': device}
        rrefs = args.encoder(LOAD_FN, kwargs, *worker_args)
        tmp = time.time()
        model, h0, tpe = train(rrefs, args, rnn_args, device)
        tr_time = time.time() - tmp
    model = model.to(device)
    h0, zs = get_cutoff(model, h0, times, args, args.fpweight, args.flows, device)
    stats = []

    for te_start,te_end in times['te_times']:
        test_times = {
            'te_start': te_start,
            'te_end': te_end,
            'delta': times['delta']
        }
        st = test(model, h0, test_times, rrefs, args.flows, device, args)
        for s in st:
            s['TPE'] = tpe
        stats += st

    pickle.dump(stats, open(OUTPATH+TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    # Retrieve stats, and cleanup temp file
    stats = pickle.load(open(OUTPATH+TMP_FILE, 'rb'))
    return stats



def train(rrefs, args, rnn_args, device):
    rnn_constructor = args.rnn
    dataset = args.dataset
    rnn = rnn_constructor(*rnn_args)
    model = Argus(rnn, rrefs, device)
    model = model.to(device)
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)
    times = []
    best = (model.save_states(), 0)
    no_progress = 0
    for e in range(args.epochs):
        # Get loss and send backward
        model.train()
        st = time.time()
        zs = model.forward(TData.TRAIN)
        loss = model.loss_fn(zs, TData.TRAIN, nratio=args.nratio, device=device, encoder_name=args.encoder_name)
        loss.backward()
        opt.step()
        elapsed = time.time()-st
        times.append(elapsed)
        l = loss.sum()
        print('[%d] Loss %0.4f  %0.2fs' % (e, l.item(), elapsed))

        # Get validation info to prevent overfitting
        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)
            auc,ap = get_score(p,n)
            print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')

            # Either incriment or update early stopping criteria
            tot = auc+ap
            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= 1:
                    no_progress += 1
            if no_progress == args.patience:
                print("Early stopping!")
                break

    model.load_states(*best[0])

    # Get the best possible h0 to eval with
    zs, h0 = model(TData.TEST, include_h=True)
    states = {'states': best[0], 'h0': h0}
    f = open('./Exps/model_save_'+dataset+'.pkl', 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)
    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)
    return model, h0, tpe


def get_cutoff(model, h0, times, args, lambda_param, use_flows, device):
    Encoder = DetectorEncoder
    ld_args = {
            'start': times['val_start'],
            'end': times['val_end'],
            'delta': times['delta'],
            'is_test': False,
            'use_flows': use_flows
        }

    Encoder.load_new_data(model.gcns, LOAD_FN, ld_args)
    # Then generate GCN embeds
    model.eval()

    zs = Encoder.forward(model.gcns.module, TData.ALL, True).to(device)
    # Finally, generate actual embeds
    with torch.no_grad():
        zs, h0 = model.rnn(zs, h0, include_h=True)

    # Then score them
    p, n = Encoder.score_edges(model.gcns, zs, TData.ALL, args.nratio)
    # Finally, figure out the optimal cutoff score
    p = p.cpu()
    n = n.cpu()
    model.cutoff = get_optimal_cutoff(p,n,fw=lambda_param)
    return h0, zs[-1]

def test(model, h0, times, rrefs, use_flows, device, args):
    Encoder = DetectorEncoder
    # Load train data into workers
    ld_args = {'start': times['te_start'],
                'end': times['te_end'],
                'delta': times['delta'],
                'is_test': True,
                'use_flows': use_flows}

    print("Loading test data")

    Encoder.load_new_data(rrefs, LOAD_FN, ld_args)
    stats = []
    model = model.to(device)
    print("Embedding Test Data...")
    test_tmp = time.time()
    with torch.no_grad():
        model.eval()
        s = time.time()
        zs = model.forward(TData.TEST, h0=h0, no_grad=True)
        ctime = time.time()-s
    # Scores all edges and matches them with name/timestamp
    scores, labels, weights = model.score_all(zs)
    test_time = time.time() - test_tmp

    stats.append(score_stats(args,scores, labels, weights, model.cutoff, ctime))
    return stats

def score_stats(args, scores, labels, weights, cutoff, ctime):
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0).clip(max=1)
    weights = np.concatenate(weights, axis=0)

    # Classify using cutoff from earlier
    classified = np.zeros(labels.shape)
    classified[scores <= cutoff] = 1

    # Calculate TPR
    p = classified[labels==1]
    tpr = p.mean()
    tp = p.sum()
    del p

    # Calculate FPR
    f = classified[labels==0]
    fp = f.sum()
    fpr = f.mean()
    del f

    cm = confusion_matrix(labels, classified, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    scores = 1-scores


    # Get metrics
    auc = auc_score(labels, scores)
    ap = ap_score(labels, scores)
    f1 = f1_score(labels, classified)

    print("Learned Cutoff %0.4f" % cutoff)
    print("TPR: %0.4f, FPR: %0.4f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))
    print("FwdTime", ctime, )
    title = "test"
    return {
        'Model': title,
        'TPR':tpr.item(),
        'FPR':fpr.item(),
        'TP':tp.item(),
        'FP':fp.item(),
        'F1':f1,
        'AUC':auc,
        'AP': ap,
        'FwdTime':ctime,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }


