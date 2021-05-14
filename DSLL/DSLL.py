# -*- coding:utf-8 -*-

# Deep Streaming Label Learning
import os
import math
import random
import numpy as np 
import pandas as pd
from sklearn import metrics
import scipy.io
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from train import   train_KD, train_DSLL_model, train_new, train_S_label_mapping, AL_train_DSLL_model
from othml import train_BR_CC
from helpers import predict, print_predict, LayerActivations, predict_integrated
from params_setting import get_params
from load_data import load_dataset, get_full_X
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

from skmultilearn.model_selection import iterative_train_test_split

from model import IntegratedDSLL, _classifier2
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CustomDataset(Dataset):
    def __init__(self, x_tensor,y_mapping_tensor, y_tensor):
        self.x = x_tensor
        self.y_mapping = y_mapping_tensor
        self.y = y_tensor
        

    def __getitem__(self, index):
        return (self.x[index], self.y_mapping[index], self.y[index])

    def __len__(self):
        return len(self.x)

def al_train_test_split(x, ymap, yold, ynew, test_percentage):
    # randomly select the indices of the train and test set
    all_indices = np.arange(0, len(x))
    random_test_indices = np.random.randint(0, len(x), int(len(x)*(test_percentage/100)))
    random_train_indices = list(set(all_indices) - set(random_test_indices))
    
    # x_train, ymap_train, x_test, ymap_test = iterative_train_test_split(x, ymap, test_size = test_percentage)
    # yold_train, ynew_train, yold_test, ynew_test = iterative_train_test_split(yold, ynew, test_size = test_percentage)
    # 
    x_train, ymap_train, yold_train, ynew_train = x[random_train_indices], ymap[random_train_indices], yold[random_train_indices], ynew[random_train_indices]
    x_test, ymap_test, yold_test, ynew_test = x[random_test_indices], ymap[random_test_indices], yold[random_test_indices], ynew[random_test_indices]

    return x_train, ymap_train, yold_train, ynew_train, x_test, ymap_test, yold_test, ynew_test



# class CustomActiveLearningDataset(Dataset):
def CustomActiveLearningDataset(x_tensor, y_mapping_tensor, old_y_tensor, y_tensor, batch_size):
    # splits the dataset in train (seed and pool) and test set
    test_percentage = 20
    x_tensor, y_mapping_tensor, old_y_tensor, y_tensor #, x_tensor_test, y_mapping_tensor_test, old_y_tensor_test, y_tensor_test = al_train_test_split(x_tensor, y_mapping_tensor, old_y_tensor, y_tensor, test_percentage)
    
    # range of all indices 
    all_indices = np.arange(0, len(x_tensor))
    
    # randomly select the seeds
    random_seed_list = np.random.randint(0, int(len(x_tensor)), int(batch_size))

    # trick to remove items from random seed list in all indices
    difference = list(set(all_indices) - set(random_seed_list))

    # select the seeds in the tensors
    xseed = x_tensor[random_seed_list]
    y_mappingseed = y_mapping_tensor[random_seed_list]
    yseed = y_tensor[random_seed_list]
    oldyseed = old_y_tensor[random_seed_list]
    # print(xseed.shape)
    # remove the seeds from the pool
    xpool = x_tensor
    y_mappingpool = y_mapping_tensor
    ypool = y_tensor
    oldypool = old_y_tensor

    # pool contains all x_tensor except for the ones in the seed
    xpool = xpool[difference]
    y_mappingpool = y_mappingpool[difference]
    ypool = ypool[difference]
    oldypool = oldypool[difference]

    # return seeds and pool
    return (xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, oldyseed, oldypool)#, (x_tensor_test, y_mapping_tensor_test, old_y_tensor_test, y_tensor_test)

def main(**kwargs):
    
    seednr = 125 #123
    random.seed(seednr)
    torch.manual_seed(seednr)
    torch.cuda.manual_seed(seednr)
    np.random.seed(seednr)
    random.seed(seednr)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    ############################################# IMPORTANT VALUES FOR DATASETS AND ACTIVE LEARNING #############################################
    # datasets = ['yeast', 'nus', 'mirfl','leda']
    # dataset = datasets[3]
    
    dataset = kwargs['dataset']
    print(dataset)
    print(kwargs)
    # use active learning yes or no
    use_al = kwargs['use_al']
    ############################################# IMPORTANT VALUES FOR DATASETS AND ACTIVE LEARNING #############################################
    
    # lower split is less old labels used, higher split is more old labels used
    split = kwargs['label_split']
    hyper_params = get_params(dataset, **kwargs)

    # with open(f'npresults/1test_yeast_{hyper_params.batch_size}_f1micro.npy', 'wb') as f:
    #     # np.save(f, np.hstack(np.array(full_measurements[f'{altype}'])[:,0]))
    #     np.save(f, [np.array([9,8,7,6,5]), np.array([1,2,3,4,5])])
    #     # np.save(f, )
    # with open(f'npresults/1test_yeast_{hyper_params.batch_size}_f1micro.npy', 'rb') as f:
    #     a = np.load(f)
    # with open(f'npresults/1test_yeast_{hyper_params.batch_size}_f1micro_aucmicro_recall.npy', 'rb') as f:
    #     a = np.load(f, allow_pickle=True)

    # print(a)
    # print("succes")
    # exit()


    # train_Y/test_Y are old y labels, test_Y_rest are new label indices. 
    train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest = load_dataset(dataset, split, hyper_params)
    
    print(f"Working with the {dataset} dataset")

    train_X_tensor = torch.from_numpy(train_X).float()
    train_Y_tensor = torch.from_numpy(train_Y).float()
    train_data = TensorDataset(train_X_tensor, train_Y_tensor)
    print('read dataset')

    hyper_params.dataset_name = dataset
    hyper_params.N = train_X.shape[0]
    hyper_params.D = train_X.shape[1]
    hyper_params.M_full = train_Y.shape[1] + train_Y_rest.shape[1]
    hyper_params.M = train_Y.shape[1]
    hyper_params.N_test = test_X.shape[0]
    hyper_params.label_mapping_input_dim = train_Y.shape[1]

    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]
    hyper_params.model_name = 'DSLL'
    hyper_params.KD_input_dim = train_X.shape[1]
    hyper_params.KD_output_dim = 200
    hyper_params.label_mapping_output_dim = train_Y.shape[1]
    hyper_params.label_representation_output_dim = train_Y.shape[1]

    if dataset == "nus":
        hyper_params.classifier_hidden1 = 512
        hyper_params.classifier_hidden2 = 256
    title1 = {dataset, 'N = {}'.format(hyper_params.N), 'D = {}'.format(hyper_params.D), 'M = {}'.format(hyper_params.M),
            'N_test = {}'.format(hyper_params.N_test)}
    print(title1)

    print('\n****************** Streaming Feature Distillation ******************\n')
    # model_old = torch.load('models/past-label-classifier')
    # import pretrained model with old labels or train it from scratch (takes a few minutes)
    if dataset == "yeast":
        try:
            print('loading past-label classifer\n')
            classifier_W_m = torch.load(
                'models/past-label-classifier-upd2').to(hyper_params.device)
        except IOError:
            print('learning past-label classifer\n')
            classifier_W_m = _classifier2(hyper_params)
            classifier_W_m = train_new(classifier_W_m, train_X, train_Y, hyper_params)
            torch.save(classifier_W_m, 'models/past-label-classifier-yeast_v02')   
    elif dataset == "leda":
        try:
            print('loading past-label classifer\n')
            classifier_W_m = torch.load(
                'models/past-label-classifier-leda_v1').to(hyper_params.device)
        except IOError:
            print('learning past-label classifer\n')
            classifier_W_m = _classifier2(hyper_params)
            classifier_W_m = train_new(classifier_W_m, train_X, train_Y, hyper_params)
    else:
        print('learning past-label classifer\n')
        classifier_W_m = _classifier2(hyper_params)
        classifier_W_m = train_new(classifier_W_m, train_X, train_Y, hyper_params)
    # torch.save(classifier_W_m, 'models/past-label-classifier-leda_v1')   

    classifier_W_m.eval()
    soft_train_Y = predict(classifier_W_m, train_X)  # sigmoid of forward pass

    # test X is predicted using the model trained on the train_x and train_y
    soft_test_Y = predict(classifier_W_m, test_X)

    # hook is meant to get the get the values from the ReLU activation functions (not the weights of the activation function)
    relu_hook_train = LayerActivations(classifier_W_m.W_m, 2)
    # relu_hook_train = LayerActivations(classifier_W_m.label_mapping, 1)
    # train_X = torch.FloatTensor(train_X).to(hyper_params.device)
    output = classifier_W_m(torch.FloatTensor(train_X).to(hyper_params.device))
    # output = classifier_W_m(train_X)
    relu_hook_train.remove()
    # values after the ReLU activation function of the train_X
    relu_out_train = relu_hook_train.features

    relu_hook_test = LayerActivations(classifier_W_m.W_m, 2)
    # test_X = torch.FloatTensor(test_X).to(hyper_params.device)
    output = classifier_W_m(torch.FloatTensor(test_X).to(hyper_params.device))
    # output = classifier_W_m(test_X)
    relu_hook_test.remove()

    # values after the ReLU activation function of test_X
    relu_out_test = relu_hook_test.features    

    # learn to predict the feature values after the relu function of classifier_W_m
    # hyper_params.KD_output_dim is same dim as the relu dimension --> hyper_params.classifier_hidden1

    print('\n****************** TRAIN KNOWLEDGE DISTILLATION ******************\n')
    if dataset == "yeast":
        try:
            print('loading past-label classifer\n')
            featureKD_model = torch.load(
                'models/knowledge_distillation-yeast_v02').to(hyper_params.device)
        except IOError:
            print('learning past-label classifer\n')
            featureKD_model = train_KD(hyper_params, train_X, relu_out_train)
            torch.save(featureKD_model, 'models/knowledge_distillation-yeast_v02')   
    else:
        featureKD_model = train_KD(hyper_params, train_X, relu_out_train) # , test_X, relu_out_test

    # Streaming Label Mapping
    print('\n****************** Streaming Label Mapping ******************\n')
    hyper_params.label_mapping_hidden1 = 200
    hyper_params.label_mapping_hidden2 = 0
    hyper_params.loss = 'correlation_aware'  # label correlation-aware loss
    device = hyper_params.device
    # test_Y_rest is unknown, only the test_Y is known
    rest_iterations = train_Y_rest.shape[1]
    if dataset == "nus":
        start_iterations = rest_iterations-5
    if dataset == "mirfl":
        start_iterations = rest_iterations-5
    if dataset == "leda":
        start_iterations = 0  
    else:
        start_iterations = 4
    for i in range(start_iterations, rest_iterations):
        print(f"New Labels number {i} from {rest_iterations}")
        # these are the to be labelled 
        train_Y_new = train_Y_rest[:, : ] #i+1
        test_Y_new = test_Y_rest[:, : ]
        
        # define shapes
        hyper_params.M_new = train_Y_new.shape[1]
        hyper_params.label_mapping_output_dim = train_Y_new.shape[1]
        hyper_params.label_representation_output_dim = train_Y_new.shape[1]
        print('apply label mapping')
        # Train_Y is available, soft train Y is predicted by the old classifier at start. Soft test Y is predicted by old class
        # model_old = torch.load('models/{}mapping'.format(i+2))
        # pepijn model:
        if dataset == "yeast" and split == 0.7:
            print("loading 0.7 split model")
            mapping_model = torch.load(
                f'models/{i+1}mapping-pep-{split}-upd', map_location=torch.device(device))
        else:
            # y is half predicted with classifier_W_m and half original y
            # the thing that is predicted are the NEW LABELS USING THE PREVIOUS LABELS
            mapping_model = train_S_label_mapping(hyper_params, 0.5 * train_Y + 0.5 * soft_train_Y, train_Y_new) # soft_test_Y
            # torch.save(mapping_model, f'models/{i+1}mapping-pep-{split}-upd')  
        
        # predicted mapping predicting the new labels using the previous labels
        mapping_train_Y_new = predict(mapping_model, 0.1 * soft_train_Y + 0.9 * train_Y)
        mapping_model.eval()
        # prediction of new test labels
        mapping_test_Y_new = predict(mapping_model,  soft_test_Y)

        # Senior Student
        mapping_train_Y_new_tensor = torch.from_numpy(mapping_train_Y_new).float()
        train_Y_new_tensor = torch.from_numpy(train_Y_new).float()
        train_Y_old_tensor = torch.from_numpy(train_Y).float()

        # if we use active leanring we cant use a loader so the whole data needs to be loaded in memory!!!!
        # possible fix is to use random set each time in loader which changes
        if use_al == True:
            seed_pool = CustomActiveLearningDataset(train_X_tensor, mapping_train_Y_new_tensor, train_Y_old_tensor,train_Y_new_tensor, hyper_params.batch_size)
            test_ds = (torch.from_numpy(test_X).float(), torch.from_numpy(mapping_test_Y_new).float(), torch.from_numpy(test_Y).float(), torch.from_numpy(test_Y_new).float())
            # print(train_X_tensor.shape ,train_Y_old_tensor.shape, train_Y_new_tensor.shape)
            # print(torch.from_numpy(test_X).float().shape, torch.from_numpy(test_Y).float().shape, torch.from_numpy(test_Y_rest).float().shape)
            # exit()
        else:
            train_data_DSLL = CustomDataset(train_X_tensor, mapping_train_Y_new_tensor, train_Y_new_tensor)
        
        hyper_params.classifier_dropout = 0.5
        hyper_params.classifier_L2 = 1e-08
        hyper_params.batchNorm = False
        hyper_params.changeloss = False
        hyper_params.loss = 'correlation_aware'   # correlation_aware  correlation_entropy  entropy
        
        # run the DSLL model with one batch size
        batch_size = hyper_params.batch_size
        
        # if true then go into the AL mode
        if use_al == True:
            
            # train_BR_CC(train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest, seed_pool, seednr)
            # CC_dataset = (train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest)
            # exit()
            myclass = AL_train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X,
                            mapping_test_Y_new, test_Y_new, seed_pool, test_ds, use_al, seednr)
                            # rain_X).float()
            # full_x = get_full_X(hyper_params).to_numpy()

            # soft_full_y = predict(classifier_W_m, full_x)

            # full_y_mapping_tensor = predict(mapping_model,  soft_full_y)

            # full_y_mapping_tensor = torch.from_numpy(full_y_mapping_tensor).float()
            # full_x_tensor = torch.from_numpy(full_x).float()

            # predictions = predict_integrated(myclass, full_x_tensor, full_y_mapping_tensor)

            # xls = pd.ExcelFile('LEDAExport_20200224_verrijkt_2_voorPepijn.xlsx')
            # df = pd.read_excel(xls, 'LEDA_20200224')
            # df['Ca'] = pd.Series(predictions.tolist())
            # pd.set_option('display.max_columns', None)  
            # pd.set_option('display.expand_frame_repr', False)
            # pd.set_option('max_colwidth', -1)
            # print(df['Ca'])
            # df = df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
            # df.to_excel("pepijn_save.xlsx")  
            # df.to_excel("pepijn_save2.xlsx", engine='xlsxwriter')
        else:    
            # hyper_params.classifier_epoch = int(40 + 1 * hyper_params.batch_size)
            train_DSLL_loader = DataLoader(dataset=train_data_DSLL,
                                        batch_size=hyper_params.batch_size,
                                        shuffle=True,
                                        num_workers=5
                                        )

            hyper_params.label_representation_hidden1 = 200
            train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X,
                            mapping_test_Y_new, test_Y_new, train_DSLL_loader, use_al)
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'DSLL active learning loss algorithm')
    parser.add_argument("--dataset", default="yeast", help="select which dataset to use, ['yeast', 'nus', 'mirfl','leda']")
    parser.add_argument("--lpm_selection", default="original", help="select which lpm learning way to use to use, ['kmeans', 'original']")
    parser.add_argument("--altype", default="lpm", help="select which active learning to use, ['worstcase', 'svm', 'random',  'rf',  'lpm']")
    parser.add_argument("--split_seed", default=25, help="select seed to get right prediction columns, 22: 11ij_6, 23: ij6_11 25: 611_ij")
    parser.add_argument("--early_stopping", default=1.0, help="How many time the full lenght we want to train")
    parser.add_argument("--label_split", default=0.5, help="Select the split percentage of labels predicted versus used in prediction higher is more labels use for prediction. (selection between 0-1, but actullay between 0.05 and 0.95 because you want at least 1 in the used and 1 in the predicted lists)")
    parser.add_argument("--batch_size", default=20, help="select the batch size (around 10-40)")
    parser.add_argument("--use_al", default=True, help="Using active learning (True, False)")
    parser.add_argument("--bsm", default=3, help="The cluster batch size multiplier (bsm) selects how many clusters are generated for the lpm AL. It is x times the batch_size so a larger batch size warrants for a lower bsm. For batch sizes 10/20 it is recommended 3 for batch size 30 it is recommended 2 for batch size 40 it is receommended 3")
    parser.add_argument("--zero_multiplier", default=3, help="(4,5,6,7) Select how many times you want to include the 0's list. The 1's list is about 1/100 of the 0's list for the first 5 vitamins/minerals so you can test with any number inbetween 1 and about 80. ")
    parser.add_argument("--pca_dimensions", default=2, help="select the pppca dimensions for the kmeans algorithm (around 3-8)")
    parser.add_argument("--classifier_lr", default=0.001, help="Selects the learning rate of the classifer")
    parser.add_argument("--lpm_lr", default=0.001, help="Selects the learning rate of the loss prediction module")
    parser.add_argument("--lpm_criterion", default='MSELOSS', help="(MSELOSS, PAIRWISE, LISTWISE) Selects the criterion (way the loss is learnt) of the loss prediction module")

    # np.random.seed(25) # from 22: 11ij_6,      23: ij6_11 25:611_ij
    # parser.add_argument("--amount", default=400_000, help="select over how many rows you want to do the unsupervised learning")
    # parser.add_argument("--nn", default=25,  help="select the amount of nn cells for the umap")
    # parser.add_argument("--min_dis", default=0.0,  help="select the minimal distance for the umap")
    # parser.add_argument("--metric", default="yule",  help="select which metric for the umap you want to compute")
    # parser.add_argument("--kmclusters", default=8, help="select how many clusters you want to look over")
    # parser.add_argument("-bestparam", action='store_true', help='calculates the best parameters for the current settings (can take hours)')
    # parser.add_argument("--bestcols", help='calculates the best columns (you need to specify which number of columns) from the datafile using the current settings for the umap (can take hours)')
    args = parser.parse_args()
    targs = args
    # args['batch_size'] = int(args['batch_size'] * args['zero_multiplier'])
classifier_lrs = [0.0001,0.0005,0.001,0.005,0.01, 0.02]
lpm_lrs = [0.0001,0.0005,0.001,0.005,0.01, 0.02]
batch_sizes = [10,15,20,25]
lpm_criterions = ["MSELOSS", "PAIRWISE", "LISTWISE"]
for clas_lr in classifier_lrs:
    for lpm_lr in lpm_lrs:
        for bz in batch_sizes:
            for lpmc in lpm_criterions:
                args = targs
                vars(args)['lpm_criterion'] = str(lpmc)
                vars(args)['classifier_lr'] = float(clas_lr)
                vars(args)['lpm_lr'] = float(lpm_lr)
                vars(args)['batch_size'] = int(bz)
                main(**vars(args))
                # print("done")
                # exit()