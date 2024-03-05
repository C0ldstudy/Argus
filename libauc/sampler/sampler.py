"""
If you use our sampler functions, please acknowledge us and cite the following papers:
@software{libauc2022,
  title={LibAUC: A Deep Learning Library for X-risk Optimization.},
  author={Zhuoning Yuan, Zi-Hao Qiu, Gang Li, Dixian Zhu, Zhishuai Guo, Quanqi Hu, Bokun Wang, Qi Qi, Yongjian Zhong, Tianbao Yang},
  year={2022}
  }
"""

import numpy as np
import random
import torch
import torchvision
from torch.utils.data.sampler import Sampler

class ImbalancedDataSampler(Sampler):
    """Base class for imabalanced dataset sampler. Support binary and multi-label tasks only. 
        
        Args:
            dataset: a customized PyTorch Dataset class 
            sampling_rate: the ratio of number of positive samples of all samples in a mini-batch (pos_sampling_rate)
            num_pos: number of postive samples in a mini-batch
            shuffle: randomly shuffling data pool at every epoch
            labels:  data labels in list/array for the given dataset. If none, obtain the labels from given dataset
        Return:
            sampled set
            
    """
    def __init__(self, 
                  dataset, 
                  batch_size=None, 
                  labels=None, 
                  shuffle=True, 
                  num_pos=None, 
                  num_tasks=None, 
                  sampling_rate=0.5): # recommended 0.5 by default.

        assert batch_size is not None, 'batch_size needs to be given!'
        assert (num_pos is None) or (sampling_rate is None), 'You can only use one of {pos_num} and {sampling_rate}!'
        if sampling_rate:
           assert sampling_rate>0.0 and sampling_rate<1.0, 'Sampling rate is out of range!'
        if labels is None:
           labels = self._get_labels(dataset)
        self.labels = self._check_labels(labels) # return: (N, ) or (N, T)
        
        self.num_samples = int(len(labels))
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        if not num_tasks:
           num_tasks = self._get_num_tasks(self.labels) 
        self.num_tasks = num_tasks
        self.pos_indices, self.neg_indices = self._get_class_index(self.labels) # task_id: 0, 1, 2, 3, ...
        self.class_counts = self._get_class_counts(self.labels) # for pos_len & neg_len 
 
        if self.sampling_rate:
           self.num_pos = int(self.sampling_rate*batch_size) 
           self.num_neg = batch_size - self.num_pos
        elif num_pos:
            self.num_pos = num_pos
            self.num_neg = batch_size - num_pos
        else:
            NotImplementedError

        # customize this number for each sampler  
        self.num_batches = len(labels)//batch_size 
        self.sampled = []

    def _get_labels(self, dataset):
      if isinstance(dataset, torch.utils.data.Dataset):
          return np.array(dataset.targets)       
      elif isinstance(dataset, torchvision.datasets.ImageFolder):
          raise NotImplementedError # TODO!
      else:
          # TODO Customized Dataset type
          raise NotImplementedError
        
    def _check_numpy_array(self, data, squeeze=True):
        if not isinstance(data, (np.ndarray, np.generic)):
           data = np.array(data)
        if squeeze:
           data = np.squeeze(data)
        return data
    
    def _check_labels(self, labels): # nan, negative, one-hot
        if np.isnan(labels).sum()>0:
           raise ValueError('NaN values in labels!') 
        labels = self._check_numpy_array(labels, squeeze=True)
        if (labels<0).sum() > 0 :
           raise ValueError('Negative values in labels') 
        if len(labels.shape) == 1:
           num_classes = np.unique(labels).size
           assert num_classes > 1, 'labels must have >=2 classes!'
           if num_classes > 2: # format multi-class to multi-label
              num_samples = len(labels)
              new_labels = np.eye(num_classes)[labels]  
              return new_labels
        return labels

    def _get_num_tasks(self, labels):
        if len(labels.shape) == 1: # binary
            return 2 
        else: 
            return labels.shape[-1] # multi-label
            
    def _get_unique_labels(self, labels):
        # binary or multi-label only
        unique_labels = np.unique(labels) if len(labels.shape)==1 else np.arange(labels.squeeze().shape[-1])
        assert len(unique_labels) > 1, 'labels must have >=2 classes!'
        return unique_labels

    def _get_class_counts(self, labels):
       # return a dict (for statistics)
       num_tasks = self._get_num_tasks(labels)
       dict = {}
       if num_tasks == 2: # binary labels have (N,1) shape
           task_id = 0
           dict[task_id] = (np.count_nonzero(labels == 1), np.count_nonzero(labels == 0) )
       else:
           task_ids = np.arange(num_tasks)              
           for task_id in task_ids:
               dict[task_id] = (np.count_nonzero(labels[:, task_id] == 1), np.count_nonzero(labels[:, task_id] == 0) )
       return dict

    def _get_class_index(self, labels, num_tasks=None):
        if not num_tasks:
           num_tasks = self._get_num_tasks(labels)
        num_tasks = num_tasks - 1 if num_tasks == 2 else num_tasks    
        pos_indices, neg_indices = {}, {}
        for task_id in range(num_tasks):
             label_t = labels[:, task_id] if num_tasks > 2 else labels
             pos_idx = np.flatnonzero(label_t==1)
             neg_idx = np.flatnonzero(label_t==0)
             np.random.shuffle(pos_idx), np.random.shuffle(neg_idx)
             pos_indices[task_id] = pos_idx
             neg_indices[task_id] = neg_idx
        return pos_indices, neg_indices
    
    def __iter__(self):
        pos_id = 0
        neg_id = 0
        # binary class
        if self.shuffle:
           np.random.shuffle(self.pos_pool)
           np.random.shuffle(self.neg_pool)
        for i in range(self.num_batches):
            for j in range(self.num_pos):
                self.sampled.append(self.pos_indices[task_id][pos_id % self.pos_len])
                pos_id += 1
            for j in range(self.num_neg):
                self.sampled.append(self.neg_indices[task_id][neg_id % self.neg_len])
                neg_id += 1    
        return iter(self.sampled)

    def __len__ (self):
        return len(self.sampled)
    
    
class DualSampler(ImbalancedDataSampler):
    """Same usage as ImbalancedDataSampler
    """
    def __init__(self, 
                  dataset, 
                  batch_size=None, 
                  labels=None, 
                  shuffle=True, 
                  num_pos=None,  
                  num_tasks=None, 
                  sampling_rate=None):
        super().__init__(dataset, batch_size, labels, shuffle, num_pos, num_tasks, sampling_rate)
        
        # sampling parameters
        assert self.num_tasks > 1, 'Labels are not binary, e.g., [0, 1]!'
        self.pos_len = self.class_counts[0][0]
        self.neg_len = self.class_counts[0][1]
        self.pos_indices, self.neg_indices = self.pos_indices[0], self.neg_indices[0]
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)
        self.num_batches = max(self.pos_len//self.num_pos, self.neg_len//self.num_neg)
        self.sampled =  [] #np.array([], dtype=np.int32)
        self.pos_ptr, self.neg_ptr = 0, 0

    def __iter__(self):
        self.sampled = [] #np.array([], dtype=np.int32)
        for i in range(self.num_batches):
            start_index = i*self.batch_size
            if self.pos_ptr+self.num_pos > self.pos_len:
                temp = self.pos_indices[self.pos_ptr:]
                np.random.shuffle(self.pos_indices)
                self.pos_ptr = (self.pos_ptr+self.num_pos)%self.pos_len
                #self.sampled = np.append(self.sampled, np.concatenate((temp,self.pos_indices[:self.pos_ptr])))  
                self.sampled.append(np.concatenate((temp, self.pos_indices[:self.pos_ptr])))
            else:
                #self.sampled = np.append(self.sampled, self.pos_indices[self.pos_ptr: self.pos_ptr+self.num_pos])
                self.sampled.append(self.pos_indices[self.pos_ptr: self.pos_ptr+self.num_pos])
                self.pos_ptr += self.num_pos

            start_index += self.num_pos
            if self.neg_ptr+self.num_neg > self.neg_len:
                temp = self.neg_indices[self.neg_ptr:]
                np.random.shuffle(self.neg_indices)
                self.neg_ptr = (self.neg_ptr+self.num_neg)%self.neg_len
                #self.sampled = np.append(self.sampled, np.concatenate((temp,self.neg_indices[:self.neg_ptr])))  
                self.sampled.append(np.concatenate((temp,self.neg_indices[:self.neg_ptr])))
            else:
                #self.sampled = np.append(self.sampled, self.neg_indices[self.neg_ptr: self.neg_ptr+self.num_neg])
                self.sampled.append(self.neg_indices[self.neg_ptr: self.neg_ptr+self.num_neg])
                self.neg_ptr += self.num_neg	

        self.sampled = np.concatenate(self.sampled)
        return iter(self.sampled)

    def __len__ (self):
        return len(self.sampled)
    
    
class TriSampler(ImbalancedDataSampler):
    """Same usage as ImbalancedDataSampler
    """
    def __init__(self, 
                  dataset, 
                  batch_size_per_task=None, #batch_size_per_task (required)
                  labels=None, 
                  shuffle=True, 
                  num_pos=None,  # overide if given      
                  num_tasks=None,  
                  sampling_rate=0.5): # (required): 0.5
        super().__init__(dataset, batch_size_per_task, labels, shuffle, num_pos, None, sampling_rate)
        
        assert self.num_tasks >=3, 'Tasks number needs to be >= 3!'
        
        # sampling parameters
        self.total_tasks = self.num_tasks # self.num_tasks denotes the total tasks
        self.batch_size_per_task = batch_size_per_task
        self.num_pos = int(self.batch_size_per_task*self.sampling_rate) if not num_pos else num_pos  
        if self.num_pos < 1:
           print('You need to set proper values for batch_size_per_task!')
           self.num_pos = 1
        self.num_neg = self.batch_size_per_task - self.num_pos 
        self.pos_len = [self.class_counts[task_id][0] for task_id in range(self.total_tasks)]
        self.neg_len = [self.class_counts[task_id][1] for task_id in range(self.total_tasks)]
        self.pos_indices, self.neg_indices = self.pos_indices, self.neg_indices
        for task_id in range(len(self.pos_indices)):
            np.random.shuffle(self.pos_indices[task_id])
            np.random.shuffle(self.neg_indices[task_id])
        # multi-task (labels)
        self.num_tasks = num_tasks if num_tasks else self.total_tasks # if num_tasks is given, then sampling num_tasks instead of using all tasks
        self.num_batches = self.labels.shape[0]//(self.batch_size_per_task*self.num_tasks)
        self.task_ptr, self.tasks_ids = 0, np.arange(self.total_tasks)
        self.pos_ptr, self.neg_ptr = np.zeros(self.total_tasks, dtype=np.int32), np.zeros(self.total_tasks, dtype=np.int32)
        self.sampled = [] #np.array([], dtype=np.int32) # prevent missing assignments
        
    def __iter__(self):
        self.sampled = [] #np.array([], dtype=np.int32)
        for i in range(self.num_batches):
            start_index = i*self.batch_size_per_task # keep it for old implementaions
            if self.num_tasks < self.total_tasks:
                task_ids = []
                # sampling tasks
                if self.task_ptr+self.num_tasks > self.total_tasks:
                    temp = self.tasks_ids[self.task_ptr:]
                    np.random.shuffle(self.tasks_ids)
                    self.task_ptr = self.task_ptr % len(self.tasks_ids)
                    task_ids = np.concatenate((temp, self.tasks_ids[:self.task_ptr]))                    
                else:
                    task_ids = self.tasks_ids[self.task_ptr: self.task_ptr+self.num_tasks]
                    self.task_ptr += self.num_tasks                      
            else:
                # use all tasks
                task_ids = self.tasks_ids
                
            for task_id in task_ids:
                if self.pos_ptr[task_id]+self.num_pos > self.pos_len[task_id]:
                    temp = self.pos_indices[task_id][self.pos_ptr[task_id]:]
                    np.random.shuffle(self.pos_indices[task_id])
                    self.pos_ptr[task_id] = (self.pos_ptr[task_id]+self.num_pos)%self.pos_len[task_id]
                    #self.sampled = np.append(self.sampled, np.concatenate((temp, self.pos_indices[task_id][:self.pos_ptr[task_id]])) )
                    self.sampled.append(np.concatenate((temp, self.pos_indices[task_id][:self.pos_ptr[task_id]])))
                else:
                    #self.sampled = np.append(self.sampled, self.pos_indices[task_id][self.pos_ptr[task_id]:self.pos_ptr[task_id]+self.num_pos])
                    self.sampled.append(self.pos_indices[task_id][self.pos_ptr[task_id]:self.pos_ptr[task_id]+self.num_pos])
                    self.pos_ptr[task_id] += self.num_pos

                start_index += self.num_pos
                if self.neg_ptr[task_id]+self.num_neg > self.neg_len[task_id]:
                    temp = self.neg_indices[task_id][self.neg_ptr[task_id]:]
                    np.random.shuffle(self.neg_indices[task_id])
                    self.neg_ptr[task_id] = (self.neg_ptr[task_id]+self.num_neg)%self.neg_len[task_id]
                    #self.sampled = np.append(self.sampled, np.concatenate((temp, self.neg_indices[task_id][:self.neg_ptr[task_id]])) )
                    self.sampled.append(np.concatenate((temp, self.neg_indices[task_id][:self.neg_ptr[task_id]])))
                else:
                    #self.sampled = np.append(self.sampled, self.neg_indices[task_id][self.neg_ptr[task_id]: self.neg_ptr[task_id]+self.num_neg])
                    self.sampled.append(self.neg_indices[task_id][self.neg_ptr[task_id]: self.neg_ptr[task_id]+self.num_neg])
                    self.neg_ptr[task_id] += self.num_neg
                start_index += self.num_neg
                
        self.sampled = np.concatenate(self.sampled)
        return iter(self.sampled)

    def __len__ (self):
        return len(self.sampled)
    

    
