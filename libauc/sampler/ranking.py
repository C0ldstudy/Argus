"""
If you use our sampler functions, please acknowledge us and cite the following papers:
 @misc{libauc2022,
      title={LibAUC: A Deep Learning Library for X-Risk Optimization.},
      author={Zhuoning Yuan, Zi-Hao Qiu, Gang Li, Dixian Zhu, Zhishuai Guo, Quanqi Hu, Bokun Wang, Qi Qi, Yongjian Zhong, Tianbao Yang},
      year={2022}
	}
"""

import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import os
from tqdm import trange

# TODO: move to sampler.py
class DataSampler(Sampler):
    """
    Data Sampler for recommender systems

    Args:
        labels: a 2-D csr sparse array: [task_num, item_num]
        batch_size: number of all labels (items) in a batch = num_tasks * (num_pos + num_neg)
        num_pos: number of positive labels (items) for each task (user)
        num_tasks: number of tasks (users)
    """
    def __init__(self, labels, batch_size, num_pos, num_tasks):
        self.labels = labels
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        
        self.num_pos = num_pos
        self.num_neg = self.batch_size//num_tasks - self.num_pos
	
        self.label_dict = {}
        
        for i in trange(self.labels.shape[0]):
            task_label = self.labels[i, :].toarray()
            pos_index = np.flatnonzero(task_label>0)
            ###To avoid sampling error
            while len(pos_index) < self.num_pos: 
                pos_index = np.concatenate((pos_index,pos_index))
            np.random.shuffle(pos_index)

            neg_index = np.flatnonzero(task_label==0)
            while len(neg_index) < self.num_neg: 
                neg_index = np.concatenate((neg_index,neg_index))
            np.random.shuffle(neg_index)

            self.label_dict.update({i:(pos_index,neg_index)})

        self.pos_ptr, self.neg_ptr = np.zeros(self.labels.shape[0], dtype=np.int32), np.zeros(self.labels.shape[0], dtype=np.int32)
        self.task_ptr, self.tasks = 0, np.random.permutation(list(range(self.labels.shape[0])))

        self.num_batches = self.labels.shape[0] // self.num_tasks

        self.sampled_task = np.empty(self.num_batches*self.num_tasks, dtype=np.int32)
        self.sampled_labels = np.empty((self.num_batches*self.num_tasks, self.num_pos+self.num_neg), dtype=np.int32)


    def __iter__(self):

        beg = 0 # beg is the pointer for self.ret

        for batch_id in range(self.num_batches):
            task_ids = self.tasks[self.task_ptr:self.task_ptr+self.num_tasks]  # randomly sample task_ids (number: self.num_tasks)
            self.task_ptr += self.num_tasks
            if self.task_ptr >= len(self.tasks):
                np.random.shuffle(self.tasks)            
                self.task_ptr = self.task_ptr % len(self.tasks)                 # if reach the end, then shuffle the list and mod the pointer
                                   
            for task_id in task_ids:
                item_list = np.empty(self.num_pos+self.num_neg, dtype=np.int16)

                if self.pos_ptr[task_id]+self.num_pos > len(self.label_dict[task_id][0]):
                    temp = self.label_dict[task_id][0][self.pos_ptr[task_id]:]
                    np.random.shuffle(self.label_dict[task_id][0])
                    self.pos_ptr[task_id] = (self.pos_ptr[task_id]+self.num_pos)%len(self.label_dict[task_id][0])
                    if self.pos_ptr[task_id]+len(temp) < self.num_pos:
                        self.pos_ptr[task_id] += self.num_pos-len(temp)
                    item_ids = np.concatenate((temp,self.label_dict[task_id][0][:self.pos_ptr[task_id]]))
                    item_list[:self.num_pos] = item_ids
                else:
                    item_ids = self.label_dict[task_id][0][self.pos_ptr[task_id]:self.pos_ptr[task_id]+self.num_pos]
                    item_list[:self.num_pos] = item_ids
                    self.pos_ptr[task_id] += self.num_pos

                if self.neg_ptr[task_id]+self.num_neg > len(self.label_dict[task_id][1]):
                    temp = self.label_dict[task_id][1][self.neg_ptr[task_id]:]
                    np.random.shuffle(self.label_dict[task_id][1])
                    self.neg_ptr[task_id] = (self.neg_ptr[task_id]+self.num_neg)%len(self.label_dict[task_id][1])
                    if self.neg_ptr[task_id]+len(temp) < self.num_neg:
                        self.neg_ptr[task_id] += self.num_neg-len(temp)
                    item_ids = np.concatenate((temp,self.label_dict[task_id][1][:self.neg_ptr[task_id]]))
                    item_list[self.num_pos:] = item_ids
                else:
                    item_ids = self.label_dict[task_id][1][self.neg_ptr[task_id]:self.neg_ptr[task_id]+self.num_neg]
                    item_list[self.num_pos:] = item_ids
                    self.neg_ptr[task_id] += self.num_neg                        # sample num_neg negative items for task_id
                
                self.sampled_task[beg] = task_id
                self.sampled_labels[beg, :] = item_list
                beg += 1

        return iter(zip(self.sampled_task, self.sampled_labels))


    def __len__ (self):
        return len(self.sampled_task)
