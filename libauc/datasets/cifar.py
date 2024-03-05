import os
import os.path
import pickle
import numpy as np
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
# reference: https://pytorch.org/vision/0.8/_modules/torchvision/datasets/cifar.html#CIFAR10

def _check_integrity(root, train_list, test_list, base_folder):
    for fentry in (train_list + test_list):
        filename, md5 = fentry[0], fentry[1]
        fpath = os.path.join(root, base_folder, filename)
        if not check_integrity(fpath, md5):
          return False
    print('Files already downloaded and verified')
    return True

def CIFAR10(root='./data', train=True):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    # download dataset 
    if not _check_integrity(root, train_list, test_list, base_folder):
       download_and_extract_archive(url=url, download_root=root, filename=filename)

    # train or test set
    if train:
      downloaded_list = train_list 
    else: 
      downloaded_list = test_list

    data = []
    targets = []
    
    # now load the picked numpy arrays
    for file_name, checksum in downloaded_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, "rb") as f:
             entry = pickle.load(f, encoding="latin1")
             data.append(entry["data"])
             if "labels" in entry:
                targets.extend(entry["labels"])
             else:
                targets.extend(entry["fine_labels"])

    # reshape data and targets
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  
    targets = np.array(targets).astype(np.int32)
    return data, targets


def CIFAR100(root='./data', train=True):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]

    # download dataset 
    if not _check_integrity(root, train_list, test_list, base_folder):
       download_and_extract_archive(url=url, download_root=root, filename=filename)

    # train or test set
    if train:
      downloaded_list = train_list 
    else: 
      downloaded_list = test_list

    data = []
    targets = []
    
    # now load the picked numpy arrays
    for file_name, checksum in downloaded_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, "rb") as f:
             entry = pickle.load(f, encoding="latin1")
             data.append(entry["data"])
             if "labels" in entry:
                targets.extend(entry["labels"])
             else:
                targets.extend(entry["fine_labels"])

    # reshape data and targets
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  
    targets = np.array(targets).astype(np.int32)
    return data, targets


if __name__ == '__main__':
    # return numpy array 
    data, targets = CIFAR10(root='./data', train=True) 
    data, targets = CIFAR100(root='./data', train=True) 
    
