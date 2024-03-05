import numpy as np


def _check_imbalance_ratio(targets):
     assert isinstance(targets, (np.ndarray, np.generic)), 'targets has to be numpy array!'
     num_samples = len(targets)
     pos_count = np.count_nonzero(targets == 1)
     neg_count = np.count_nonzero(targets == 0) # check if negative labels in dataset
     pos_ratio = pos_count/ (pos_count + neg_count)
     print ('#SAMPLES: [%d], POS:NEG: [%d : %d], POS RATIO: %.4f'%(num_samples, pos_count, neg_count, pos_ratio) )

def _check_array_type(arr):
     assert isinstance(arr, (np.ndarray, np.generic)), 'Inputs need to be numpy array type!'          
          
          
class ImbalancedDataGenerator(object):
     '''
     Binary, Numpy array only 
     Added support for dataset type imbalance modififcation??? 
     '''
     def __init__(self, imratio=None, shuffle=True, random_seed=0, verbose=False):
        self.imratio = imratio # for testing set, use 0.5 instead of is_balanced
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.verbose = verbose
    
     def _get_split_index(self, num_classes):
          if num_classes == 2: 
             split_index = 0 
          elif num_classes == 10: 
             split_index = 4     
          elif num_classes == 100:
             split_index = 49 
          elif num_classes == 1000:
             split_index = 499      
          else:
             raise NotImplementedError
          return split_index

     def _get_class_num(self, targets):
          return np.unique(targets).size
     
     def transform(self, data, targets, imratio=None):
          _check_array_type(data)
          _check_array_type(targets)
          if min(targets) < 0: # check negative values
             targets[targets<0] = 0
          if imratio is not None:
             self.imratio = imratio
          assert self.imratio>0 and  self.imratio<=0.5, 'imratio needs to be in (0, 0.5)!'
          
          # shuffle once and create data copies
          id_list = list(range(targets.shape[0]))
          np.random.seed(self.random_seed)
          np.random.shuffle(id_list)
          data_copy = data[id_list].copy()
          targets_copy = targets[id_list].copy()
               
          # make binary dataset
          num_classes = self._get_class_num(targets)
          split_index = self._get_split_index(num_classes)
          targets_copy[targets_copy<=split_index] = 0 # [0, ....]
          targets_copy[targets_copy>=split_index+1] = 1 # [0, ....]

          # randomly remove some samples
          if self.imratio < 0.5:
              num_neg = np.where(targets_copy==0)[0].shape[0]
              num_pos = np.where(targets_copy==1)[0].shape[0]
              keep_num_pos = int((self.imratio/(1-self.imratio))*num_neg )
              neg_id_list = np.where(targets_copy==0)[0] 
              pos_id_list = np.where(targets_copy==1)[0][:keep_num_pos] 
              data_copy = data_copy[neg_id_list.tolist() + pos_id_list.tolist() ] 
              targets_copy = targets_copy[neg_id_list.tolist() + pos_id_list.tolist() ]
              targets_copy = targets_copy.reshape(-1, 1).astype(float)

          if self.shuffle:
             # shuffle in case batch prediction error
             id_list = list(range(targets_copy.shape[0]))
             np.random.seed(self.random_seed)
             np.random.shuffle(id_list)
             data_copy = data_copy[id_list]
             targets_copy = targets_copy[id_list]

          if self.verbose:
             _check_imbalance_ratio(targets_copy)

          return data_copy, targets_copy
