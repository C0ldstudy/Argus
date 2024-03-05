import os
import pickle
from tqdm import tqdm

# Set dataset paths
# =============
RED = '' # Location of redteam.txt
SRC = '' # Location of auth.txt
DST = '' # Directory to save output files to
# =============

assert RED and SRC and DST, 'Please download the LANL data set, and mark in the code where it is'


DELTA = 10000
DAY = 60**2 * 24

def mark_anoms():
    with open(RED, 'r') as f:
        red_events = f.read().split()
    red_events = red_events[1:]

    def add_ts(d, val, ts):
        val = (val[1], val[2])
        if val in d:
            d[val].append(ts)
        else:
            d[val] = [ts]

    anom_dict = {}
    for event in red_events:
        tokens = event.split(',')
        ts = int(tokens.pop(0))
        add_ts(anom_dict, tokens, ts)

    return anom_dict


def mark_anoms_node():
    with open(RED, 'r') as f:
        red_events = f.read().split()

    # Slice out header
    red_events = red_events[1:]

    def add_ts(d, val, ts):
        if val[1] in d:
            d[val[1]].append(ts)
        else:
            d[val[1]] = [ts]
        if val[2] in d:
            d[val[2]].append(ts)
        else:
            d[val[2]] = [ts]

    anom_dict = {}
    for event in red_events:
        tokens = event.split(',')
        ts = int(tokens.pop(0))
        add_ts(anom_dict, tokens, ts)

    return anom_dict


def is_anomalous(d, src, dst, ts):
    if ts < 573290 or (src, dst) not in d:
        return False
    times = d[(src,dst)]
    for time in times:
        if ts == time:
            return True
    return False

def is_anomalous_range(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False

    times = d[(src,dst)]
    for time in times:
        # Mark true if node appeared in a compromise in -/5min
        if abs(ts-time) <= 300:
            return True
    return False

def is_anomalous_node_range(d, node, ts):
    if ts < 150885 or node not in d:
        return False

    times = d[node]
    for time in times:
        # Mark true if node appeared in a compromise in -/5min
        if abs(ts-time) <= 300:
            return True

    return False

def save_map(m, fname):
    m_rev = [None] * (max(m.values()) + 1)
    for (k,v) in m.items():
        m_rev[v] = k

    with open(DST + fname, 'wb') as f:
        pickle.dump(m_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(DST + fname + ' saved')

def get_or_add(n, m, id):
    if n not in m:
        m[n] = id[0]
        id[0] += 1

    return m[n]


def split_auth():
    anom_dict = mark_anoms()

    last_time = 1
    cur_time = 0

    f_in = open(SRC,'r')
    f_out = open(DST + str(cur_time) + '.txt', 'w')

    line = f_in.readline() # Skip headers
    line = f_in.readline()

    nmap = {}
    nid = [0]
    prog = tqdm(desc='Seconds parsed', total=757648)

    fmt_src = lambda x : \
        x.split('@')[0].replace('$', '')

    fmt_label = lambda ts,src,dst : \
        1 if is_anomalous(anom_dict, src, dst, ts) \
        else 0

    # ['timestamps', 'source', 'target', 'label',  'pid', 'ppid', 'dest_port', 'l4protocol', 'img_path']
    fmt_line = lambda ts, src, dst, label, pid, ppid, dest_port, l4protocol, img_path: (
        '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
            ts, get_or_add(src, nmap, nid), get_or_add(dst, nmap, nid),
            pid, ppid, dest_port, l4protocol, img_path, label),
        int(ts)
    )

    while line:
        tokens = line.split(',')
        l, ts = fmt_line(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7], tokens[8][:-1])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        # After ts progresses at least 10,000 seconds, make a new file
        if ts >= cur_time+DELTA:
            f_out.close()
            cur_time += DELTA
            f_out = open(DST + str(cur_time) + '.txt', 'w')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(nmap, 'nmap.pkl')

def reverse_load_map(fname):
    m = {}

    with open(DST+fname, 'rb') as f:
        l = pickle.load(f)
        for i in range(0, len(l)):
            m[l[i]] = i
    return m

if __name__ == '__main__':
    split_auth()
