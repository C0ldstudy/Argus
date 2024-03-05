from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np

def auroc(labels, scores, **kwargs):
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(scores, list):
        scores = np.array(scores) 
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    if len(scores.shape) == 1:
        scores = scores.reshape(-1, 1) 
    if scores.shape[-1] != 1 and len(scores.shape)>1:
        class_auc_list = []
        for i in range(scores.shape[-1]):
            try:
                local_auc = roc_auc_score(labels[:, i], scores[:, i],  **kwargs)
                class_auc_list.append(local_auc)
            except: 
                class_auc_list.append(0.0) # if only one class
        return class_auc_list
    return roc_auc_score(labels, scores, **kwargs)

def auprc(labels, scores, **kwargs):
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(scores, list):
        scores = np.array(scores)  
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    if len(scores.shape) == 1:
        scores = scores.reshape(-1, 1)    
    if scores.shape[-1] != 1 and len(scores.shape)>1:
        class_auc_list = []
        for i in range(scores.shape[-1]):
            try:
                local_auc = average_precision_score(labels[:, i], scores[:, i])
                class_auc_list.append(local_auc)
            except: 
                class_auc_list.append(0.0)
        return class_auc_list
    return average_precision_score(labels, scores)

def pauc(labels, scores, max_fpr=1.0, min_tpr=0.0, **kwargs):
    # multi-task support: TBD
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(scores, list):
        scores = np.array(scores)   
    labels = labels.reshape(-1)
    scores = scores.reshape(-1)
    if min_tpr == 0:
        # one-way partial AUC
        return roc_auc_score(labels, scores, max_fpr=max_fpr)
    # two-way partial AUC
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels != 1)[0]
    num_pos = round(len(pos_idx)*(1-min_tpr))
    num_neg = round(len(neg_idx)*max_fpr)
    num_pos = 1 if num_pos < 1 else num_pos
    num_neg = 1 if num_neg < 1 else num_neg
    if len(pos_idx)==1: 
        selected_pos = [0]
    else:
        selected_pos = np.argpartition(scores[pos_idx], num_pos)[:num_pos]
    if len(neg_idx)==1: 
        selected_neg = [0]
    else:
        selected_neg = np.argpartition(-scores[neg_idx], num_neg)[:num_neg]
    selected_target = np.concatenate((labels[pos_idx][selected_pos], labels[neg_idx][selected_neg]))
    selected_pred = np.concatenate((scores[pos_idx][selected_pos], scores[neg_idx][selected_neg]))
    return roc_auc_score(selected_target, selected_pred)

# TODO: make individual function
def map_at_k(hit, gt_rank):
        ap_list = []
        hit_gt_rank = (hit * gt_rank).astype(float)
        sorted_hit_gt_rank = np.sort(hit_gt_rank)
        for idx, row in enumerate(sorted_hit_gt_rank):
            precision_list = []
            counter = 1
            for item in row:
                if item > 0:
                    precision_list.append(counter / item)
                    counter += 1
            ap = np.sum(precision_list) / np.sum(hit[idx]) if np.sum(hit[idx]) > 0 else 0
            ap_list.append(ap)
        return np.mean(ap_list)

# TODO: make individual function
def ndcg_at_k(ratings, normalizer_mat, hit, gt_rank, k):
    # calculate the normalizer first
    normalizer = np.sum(normalizer_mat[:, :k], axis=1)
    # calculate DCG
    DCG = np.sum(((np.exp2(ratings) - 1) / np.log2(gt_rank+1)) * hit.astype(float), axis=1)
    return np.mean(DCG / normalizer)

# alias
auc_roc_score = auroc
auc_prc_score = auprc
pauc_roc_score = pauc

if __name__ == '__main__':
    # import numpy as np
    preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    # print (preds.shape, labels.shape)
    print (auprc(labels, preds))
    print (auroc(labels, preds))
    
    print (roc_auc_score(labels, preds))
    print (average_precision_score(labels, preds))


