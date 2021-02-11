import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import string
from scipy.io import arff


import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

import pandas as pd
from sklearn.utils import Bunch


from .args import args

from PIL import ImageFont, ImageDraw, Image

# temp
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import hamming_loss, average_precision_score, precision_score



# ----- Data to Image Transformer -----
# FONT_HERSHEY_PLAIN, 
def data2img(arr, font_size=50, resolution=(256, 256), font=cv2.FONT_HERSHEY_SIMPLEX): # SIMPLEX
    """ Structured Tabular Data to Image with cv2

        NOTE currently supports only iris, wine and womanshealth dataset
    """
    x, y = resolution
    if args.dataset=='mltoy':
        n_colums, n_features = 17, len(arr)
    elif args.dataset =="yeast14c":
        n_colums, n_features = 4, len(arr)
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
                        fontFace=font, fontScale=0.5, color=(255, 255, 255), thickness=1)
                    k += 1
                except IndexError:
                    break
    
    # ----- toy -----
    elif args.dataset=='yeast14c':
        # font = ImageFont.truetype("src/utils/arial-unicode-ms.ttf", 28, encoding="unic")
        # font = cv2.FONT_ITALIC
        n_lines-=5 #extra
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (5+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.32, color=(255, 255, 255), thickness=1)
                    k += 1
                except IndexError:
                    break
    # font 0.3 lines 5 SIMPLEX round 5

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
            # try:
            #     x[43:] = self.le.transform(x[43:])
            #     x[2] = self.le.transform([x[2]])[0]
            # except:
            #     pass
            img = np.array([x]).astype(dtype = 'float32')
            x = torch.from_numpy(img)
            
        if self.transform and self.make_img == True:
            x = self.transform(img)
        try:
            y = self.data[1].loc[index]
        except:
            y = self.data[1][index]
        # y = self.data[1].loc[index]
        # print(y)
        return x, y

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
def classical_predictors(x_train, y_train, x_val, y_val):
    # xgboost predictions
    xgboost_est = XGBClassifier(eval_metric='map',objective='binary:logistic',n_jobs=-1, max_depth=4, use_label_encoder=False)
    clf = OneVsRestClassifier(xgboost_est)
    print(x_train.shape, y_train.shape)
    clf.fit(x_train,y_train)
    print("xgboost results:")
    print("Hamming Loss: ",hamming_loss(y_val,clf.predict(x_val)))
    print("avg precsion: ",average_precision_score(y_val,clf.predict(x_val)))
    print("precision mi: ",precision_score(y_val,clf.predict(x_val),average='micro'))
    
    # randomforest predictions
    clf = RandomForestClassifier(max_depth=4, random_state=0)
    clf.fit(x_train, y_train)
    
    print("\nRandom Forest results:")
    print("Hamming Loss: ",hamming_loss(y_val,clf.predict(x_val)))
    print("avg precsion: ",average_precision_score(y_val,clf.predict(x_val)))
    print("precision mi: ",precision_score(y_val,clf.predict(x_val),average='micro'))
    exit()

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
    elif dataset == "yeast14c":
        data = pd.read_csv("src/utils/ml/yeast_14class.csv")
        xtrain = data.iloc[:, :len(list(data))-14]
        xtrain = xtrain.to_numpy().round(8)
        # print(xtrain.shape)
        
        ytrain = data.iloc[:, len(list(data))-14:]
        ytrain = ytrain.to_numpy()
        data = Bunch(data=xtrain, target=ytrain)
    elif dataset == "yeast14c_m":

        train_df = arff.loadarff('src/utils/ml/mulan_yeast/yeast-train.arff')
        train_df = pd.DataFrame(train_df[0])
        # train_df.reset_index(inplace=True)
        x_train = train_df.iloc[:, :len(list(train_df))-14]
        x_train = x_train.to_numpy().round(8)
        y_train = train_df.iloc[:, len(list(train_df))-14:].to_numpy().astype(str).astype(int)

        test_df = arff.loadarff('src/utils/ml/mulan_yeast/yeast-test.arff')
        test_df = pd.DataFrame(test_df[0])
        # test_df.reset_index(inplace=True)
        x_test = test_df.iloc[:, :len(list(test_df))-14]
        x_test = x_test.to_numpy().round(8)
        y_test = test_df.iloc[:, len(list(test_df))-14:].to_numpy().astype(str).astype(int)



    if dataset != "yeast14c_m":
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

    # classical_predictors(x_train, y_train, x_val, y_val)
    
    make_img = False if args.model == 'no_img' else True
    # Build Dataset
    train_data = CustomTensorDataset(data=(x_train, y_train), transform=transform, make_img=make_img)
    val_data   = CustomTensorDataset(data=(x_val, y_val), transform=transform, make_img=make_img)
    test_data  = CustomTensorDataset(data=(x_test, y_test), transform=transform, make_img=make_img)

    # Build Dataloader
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_data, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_data, shuffle=False, **loader_kwargs)
    # 3, 256, 256
    #return train_data, val_data, test_data
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    pass

