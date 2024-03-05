import torch

def edge_tvt_split(ei):
    ne = ei.size(1)
    val = int(ne*0.85)
    te = int(ne*0.90)

    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)
    masks[0, rnd[:val]] = True
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True

    return masks[0], masks[1], masks[2]

def edge_tv_split(ei, v_size=0.05):
    ne = ei.size(1)
    val = int(ne*v_size)

    masks = torch.zeros(2, ne).bool()
    rnd = torch.randperm(ne)
    masks[1, rnd[:val]] = True
    masks[0, rnd[val:]] = True

    return masks[0], masks[1]

'''
Various weighting functions for edges
'''
def std_edge_w(ew_ts):
    ews = []
    # print('ew_ts: ', len(ew_ts))
    # print(len(ew_ts[0]))
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        ew_t = (ew_t.long() / ew_t.std()).long()
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews
def std_edge_a(ea_ts):
    eas = []

    for ea_t in ea_ts:
        ea_t2 = torch.empty(ea_t.size())
        for i in range(0, len(ea_t)):
            ea_t_f = ea_t[i]
            ea_t_f = ea_t_f.float()
            ea_t_f = (ea_t_f.long() / ea_t_f.std()).long()
            ea_t_f = torch.sigmoid(ea_t_f)
            ea_t2[i] = ea_t_f
        eas.append(ea_t2)
    return eas


def normalized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        ew_t = ew_t.true_divide(ew_t.mean())
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)
    return ews

def standardized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        std = ew_t.std()
        if std.item() == 0:
            ews.append(torch.full(ew_t.size(), 0.5))
            continue

        ew_t = (ew_t - ew_t.mean()) / std
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews

def inv_standardized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        ew_t = (ew_t - ew_t.mean()) / ew_t.std()
        ew_t = 1-torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews
