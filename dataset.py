import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os.path
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms

#==============================================================
# Custom dataset prepration in Pytorch format

class SkinData(Dataset):
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

#=============================================================================

#                         Data loading 

#============================================================================= 

def prepare_dataset(num_users, dataset, loader) :
    df = pd.read_csv('data/HAM10000_metadata.csv')
    print(df.head())

    lesion_type = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
    imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join("data", '*', '*.jpg'))}


    #print("path---------------------------------------", imageid_path.get)
    df['path'] = df['image_id'].map(imageid_path.get)
    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['target'] = pd.Categorical(df['cell_type']).codes
    print(df['cell_type'].value_counts())
    print(df['target'].value_counts())

    #=============================================================================
    # Train-test split          
    train, test = train_test_split(df, test_size = 0.2)

    train = train.reset_index(); test = test.reset_index()

    #=============================================================================
    #                         Data preprocessing
    #=============================================================================  
    # Data preprocessing: Transformation 
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                            transforms.RandomVerticalFlip(),
                            transforms.Pad(3),
                            transforms.RandomRotation(10),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])
        
    test_transforms = transforms.Compose([
                            transforms.Pad(3),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])    


    # With augmentation
    dataset_train = SkinData(train, transform = train_transforms)
    dataset_test = SkinData(test, transform = test_transforms)

    #----------------------------------------------------------------
    return dataset_train, dataset_test, dataset_iid(dataset_train, num_users), dataset_iid(dataset_test, num_users)
