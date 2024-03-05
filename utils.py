import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, \
    roc_curve, precision_recall_curve, auc, f1_score
import torch
import os

def get_score(nscore, pscore):
    ntp = pscore.size(0)
    ntn = nscore.size(0)

    score = (1-torch.cat([pscore.detach(), nscore.detach()])).cpu().numpy()
    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return [auc, ap]

def get_auprc(probs, y):
    p, r, _ = precision_recall_curve(y, probs)
    pr_curve = auc(r,p)
    return pr_curve

def tf_auprc(t, f):
    nt = t.size(0)
    nf = f.size(0)

    y_hat = torch.cat([t,f], dim=0)
    y = torch.zeros((nt+nf,1))
    y[:nt] = 1

    return get_auprc(y_hat, y)

def get_f1(y_hat, y):
    return f1_score(y, y_hat)

def get_optimal_cutoff(pscore, nscore, fw=0.5):
    ntp = pscore.size(0)
    ntn = nscore.size(0)

    tw = 1-fw

    score = torch.cat([pscore.detach(), nscore.detach()]).numpy()
    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    fpr, tpr, th = roc_curve(labels, score)
    fn = np.abs(tw*tpr-fw*(1-fpr))
    best = np.argmin(fn, 0)

    print("Optimal cutoff %0.4f achieves TPR: %0.2f FPR: %0.2f on train data"
        % (th[best], tpr[best], fpr[best]))
    return th[best]
