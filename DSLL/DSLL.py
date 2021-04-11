# -*- coding:utf-8 -*-

# Deep Streaming Label Learning
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from train import   train_KD, train_DSLL_model, train_new, train_S_label_mapping, AL_train_DSLL_model
from othml import train_BR_CC
from helpers import predict, print_predict, LayerActivations
from params_setting import get_params
from load_data import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

from model import IntegratedDSLL, _classifier2
import random
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, x_tensor,y_mapping_tensor, y_tensor):
        self.x = x_tensor
        self.y_mapping = y_mapping_tensor
        self.y = y_tensor
        

    def __getitem__(self, index):
        return (self.x[index], self.y_mapping[index], self.y[index])

    def __len__(self):
        return len(self.x)

# class CustomActiveLearningDataset(Dataset):
def CustomActiveLearningDataset(x_tensor,y_mapping_tensor, old_y_tensor, y_tensor, seeds = 10):
    # randomly select the seeds
    all_indices = np.arange(0, len(x_tensor))
    random_seed_list = np.random.randint(0, len(x_tensor), seeds)

    # trick to remove items from random seed list in all indices
    difference = list(set(all_indices) - set(random_seed_list))

    # select the seeds in the tensors
    xseed = x_tensor[random_seed_list]
    y_mappingseed = y_mapping_tensor[random_seed_list]
    yseed = y_tensor[random_seed_list]
    oldyseed = old_y_tensor[random_seed_list]

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

    # train_test_split(xpool, ypool)

    # return seeds and pool
    return (xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, oldyseed, oldypool)


if __name__ == '__main__':
    seednr = 126 #123
    random.seed(seednr)
    torch.manual_seed(seednr)
    torch.cuda.manual_seed(seednr)
    np.random.seed(seednr)
    random.seed(seednr)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    ############################################# IMPORTANT VALUES FOR DATASETS AND ACTIVE LEARNING #############################################
    datasets = ['yeast', 'nus', 'mirfl']
    dataset = datasets[0]
    use_al = True
    ############################################# IMPORTANT VALUES FOR DATASETS AND ACTIVE LEARNING #############################################
    
    # lower split is less old labels used, higher split is more old labels used
    split=0.5

    # train_Y/test_Y are old y labels, test_Y_rest are new label indices. 
    train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest = load_dataset(dataset, split)
    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_X.shape)
    # exit()
    hyper_params = get_params(dataset)

    train_X_tensor = torch.from_numpy(train_X).float()
    train_Y_tensor = torch.from_numpy(train_Y).float()
    train_data = TensorDataset(train_X_tensor, train_Y_tensor)
    print('have read dataset')

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


    # Streaming Label Distillation
    print('\n****************** Streaming Feature Distillation ******************\n')
    print('load past-label classifer\n')
    # model_old = torch.load('models/past-label-classifier')
    # yeast model pretrained
    if dataset == "yeast":
        classifier_W_m = torch.load(
            'models/past-label-classifier-upd2').to(hyper_params.device)
    else:
        classifier_W_m = _classifier2(hyper_params)
        classifier_W_m = train_new(classifier_W_m, train_X, train_Y)
    # torch.save(classifier_W_m, 'models/past-label-classifier-upd3')   

    classifier_W_m.eval()
    soft_train_Y = predict(classifier_W_m, train_X)  # sigmoid of forward pass
    # print(soft_train_Y.shape)
    # exit()
    # old model prediction output are predicted labels of old model
    soft_test_Y = predict(classifier_W_m, test_X)
    print("soft test shape")
    print(soft_test_Y.shape)


    relu_hook_train = LayerActivations(classifier_W_m.W_m, 2)
    # relu_hook_train = LayerActivations(classifier_W_m.label_mapping, 1)
    # train_X = torch.FloatTensor(train_X).to(hyper_params.device)
    output = classifier_W_m(torch.FloatTensor(train_X).to(hyper_params.device))
    # output = classifier_W_m(train_X)
    relu_hook_train.remove()
    relu_out_train = relu_hook_train.features
    # print(relu_hook_train)
    # exit()

    relu_hook_test = LayerActivations(classifier_W_m.W_m, 2)
    # test_X = torch.FloatTensor(test_X).to(hyper_params.device)
    output = classifier_W_m(torch.FloatTensor(test_X).to(hyper_params.device))
    # output = classifier_W_m(test_X)
    relu_hook_test.remove()
    relu_out_test = relu_hook_test.features

    hyper_params.KD_epoch = 1

    # knowledge destillation from teacher to new model
    featureKD_model = train_KD(hyper_params, train_X, relu_out_train, test_X, relu_out_test)

    # train_Y and test_Y we know
    # featureKD_model = train_KD(hyper_params, train_X, train_Y, test_X, test_Y)

    # Streaming Label Mapping
    print('\n****************** Streaming Label Mapping ******************\n')
    hyper_params.label_mapping_hidden1 = 200
    hyper_params.label_mapping_hidden2 = 0
    hyper_params.loss = 'correlation_aware'  # label correlation-aware loss

    # test_Y_rest is unknown, only the test_Y is known
    rest_iterations = train_Y_rest.shape[1]
    if dataset == "nus":
        start_iterations = rest_iterations-5
    if dataset == "mirfl":
        start_iterations = rest_iterations-5
    else:
        start_iterations = 4
    for i in range(start_iterations, rest_iterations):
        print(f"New Labels number {i} from {rest_iterations}")
        # these are the to be labelled 
        train_Y_new = train_Y_rest[:, :i+1]
        test_Y_new = test_Y_rest[:, :i+1]
        
        # define shapes
        hyper_params.M_new = train_Y_new.shape[1]
        hyper_params.label_mapping_output_dim = train_Y_new.shape[1]
        hyper_params.label_representation_output_dim = train_Y_new.shape[1]
        print('apply label mapping')
        # Train_Y is available, soft train Y is predicted by the old classifier at start. Soft test Y is predicted by old class
        # model_old = torch.load('models/{}mapping'.format(i+2))
        # pepijn model:
        if dataset == "yeast":
            mapping_model = torch.load(
                f'models/{i+1}mapping-pep-{split}-upd')
        else:
            # y is half predicted and half original
            mapping_model = train_S_label_mapping(hyper_params, 0.5 * train_Y + 0.5 * soft_train_Y, train_Y_new) # soft_test_Y
            # torch.save(mapping_model, f'models/{i+1}mapping-pep-{split}-upd')  
        mapping_train_Y_new = predict(mapping_model, 0.1 * soft_train_Y + 0.9 * train_Y)
        mapping_model.eval()
        mapping_test_Y_new = predict(mapping_model,  soft_test_Y)

        # Senior Student
        mapping_train_Y_new_tensor = torch.from_numpy(mapping_train_Y_new).float()
        train_Y_new_tensor = torch.from_numpy(train_Y_new).float()
        train_Y_old_tensor = torch.from_numpy(train_Y).float()

        # if we use active leanring we cant use a loader so the whole data needs to be loaded in memory!!!!
        # possible fix is to use random set each time in loader which changes
        if use_al == True:
            batch_size = 10
            seed_pool = CustomActiveLearningDataset(train_X_tensor, mapping_train_Y_new_tensor, train_Y_old_tensor,train_Y_new_tensor, batch_size)
        else:
            train_data_DSLL = CustomDataset(train_X_tensor, mapping_train_Y_new_tensor, train_Y_new_tensor)
        
        hyper_params.classifier_dropout = 0.5
        hyper_params.classifier_L2 = 1e-08
        hyper_params.batchNorm = False
        hyper_params.changeloss = False
        hyper_params.loss = 'correlation_aware'   # correlation_aware  correlation_entropy  entropy

        # 5, 10, 15, 20
        for batch_size in [10]:
            hyper_params.batch_size = batch_size

            # if true then go into the AL mode
            if use_al == True:
                
                # train_BR_CC(train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest, seed_pool, seednr)
                # CC_dataset = (train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest)
                # exit()
                AL_train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X,
                                mapping_test_Y_new, test_Y_new, seed_pool, use_al, seednr)
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

