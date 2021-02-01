import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import string

import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

import pandas as pd
from sklearn.utils import Bunch


from .args import args



# ----- Data to Image Transformer -----

def data2img(arr, font_size=50, resolution=(256, 256), font=cv2.FONT_HERSHEY_SIMPLEX):
    """ Structured Tabular Data to Image with cv2

        NOTE currently supports only iris, wine and womanshealth dataset
    """
    x, y = resolution
    if args.dataset=='mltoy':
        n_colums, n_features = 17, len(arr)
    else:
        n_colums, n_features = 2, len(arr)
    n_lines = n_features % n_colums + int(n_features / n_colums)
    frame = np.ones((*resolution, 3), np.uint8)*0

    k = 0
    # ----- iris -----
    if args.dataset=='iris':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=1, color=(255, 255, 255), thickness=2)
                    k += 1
                except IndexError:
                    break

    # ----- wine -----
    elif args.dataset=='wine':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.6, color=(255, 255, 255), thickness=1)
                    k += 1
                except IndexError:
                    break

    # ----- toy -----
    elif args.dataset=='mltoy':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (5+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.3, color=(255, 255, 255), thickness=1)
                    k += 1
                except IndexError:
                    break


    return np.array(frame, np.uint8)


# ----- Dataset -----

class CustomTensorDataset(Dataset):
    def __init__(self, data, transform=None, make_img=True):
        self.data = data
        self.transform = transform
        self.make_img = make_img
        self.le = preprocessing.LabelEncoder()
        self.le.fit(list(string.ascii_lowercase)+list(string.ascii_uppercase))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        try:
            x = self.data[0].loc[index]
        except:
            x = self.data[0][index]
        if self.make_img:
            img = data2img(x)
        else:
            # print(x)
            try:
                x[43:] = self.le.transform(x[43:])
                x[2] = self.le.transform([x[2]])[0]
            except:
                pass
            img = np.array([x]).astype(dtype = 'float32')
            x = torch.from_numpy(img)
            
        # if self.make_img == True:
        # 
        if self.transform:
            x = self.transform(img)
        try:
            y = self.data[1].loc[index]
        except:
            y = self.data[1][index]
        return x, y



# ----- Load Data Pipeline -----

import os

def load_data(dataset=args.dataset, batch_size=args.batch_size, val_size=args.val_size, test_size=args.test_size, device='cpu'):
    # load dataset
    if dataset=='iris':
        data = datasets.load_iris()
    elif dataset=='wine':
        data = datasets.load_wine()
    elif dataset=='mltoy':
        xtrain = pd.read_csv("src/utils/ml/X_train_RE.csv")
        xtrain.reset_index(inplace=True)
        xtrain = xtrain[:int(len(xtrain)/50)]
        xtrain = xtrain.to_numpy()

        ytrain = pd.read_csv("src/utils/ml/y_train_RE.csv")
        ytrain = ytrain.drop(ytrain.columns[[0]], axis=1) 
        ytrain.reset_index(inplace=True)
        ytrain = ytrain[:int(len(ytrain)/50)]
        ytrain = ytrain.to_numpy()
        ytrain = ytrain[..., 1:] # remove first index element from each row

        data = Bunch(data=xtrain, target=ytrain)

    
    # Split dataset -- Cross Vaidation
    x_train, x_test, y_train, y_test \
        = train_test_split(data.data, data.target, test_size=test_size, random_state=1)

    x_train, x_val, y_train, y_val \
        = train_test_split(x_train, y_train, test_size=val_size, random_state=1)

    

    # Dataset and Dataloader settings
    kwargs = {} if args.device=='cpu' else {'num_workers': 2, 'pin_memory': True}
    loader_kwargs = {'batch_size':batch_size, **kwargs}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    # Build Dataset
    train_data = CustomTensorDataset(data=(x_train, y_train), transform=transform, make_img=True)
    val_data   = CustomTensorDataset(data=(x_val, y_val), transform=transform, make_img=True)
    test_data  = CustomTensorDataset(data=(x_test, y_test), transform=transform)

    # Build Dataloader
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_data, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_data, shuffle=False, **loader_kwargs)
    # 3, 256, 256
    #return train_data, val_data, test_data
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    pass

