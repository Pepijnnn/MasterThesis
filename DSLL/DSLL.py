# -*- coding:utf-8 -*-

# Deep Streaming Label Learning

'''
Deep Streaming Label Learning

This code is temporarily confidential
'''
#print(__doc__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from train import train_S_label_mapping,  train_KD, train_DSLL_model, train_new
from helpers import predict, print_predict, LayerActivations
from params_setting import get_params
from load_data import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

from model import KnowledgeDistillation_start, IntegratedDSLL, _classifier2
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

if __name__ == '__main__':
    seednr = 123
    random.seed(seednr)
    torch.manual_seed(seednr)
    torch.cuda.manual_seed(seednr)
    np.random.seed(seednr)
    random.seed(seednr)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    dataset = "yeast"
    split=0.5
    train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest = load_dataset(dataset, split)
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
    title1 = {dataset, 'N = {}'.format(hyper_params.N), 'D = {}'.format(hyper_params.D), 'M = {}'.format(hyper_params.M),
            'N_test = {}'.format(hyper_params.N_test)}
    print(title1)


    # Streaming Label Distillation
    print('\n****************** Streaming Feature Distillation ******************\n')
    print('load past-label classifer\n')
    # model_old = torch.load('models/past-label-classifier')
    classifier_W_m = torch.load(
        'models/past-label-classifier-upd2').to(hyper_params.device)
    # classifier_W_m = _classifier2(hyper_params)
    # classifier_W_m = train_new(classifier_W_m, train_X, train_Y)
    # torch.save(classifier_W_m, 'models/past-label-classifier-upd2')   

    classifier_W_m.eval()
    soft_train_Y = predict(classifier_W_m, train_X)  # sigmoid
    # print(soft_train_Y.shape)
    # exit()
    soft_test_Y = predict(classifier_W_m, test_X)


    relu_hook_train = LayerActivations(classifier_W_m.W_m, 2)
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
    featureKD_model = train_KD(hyper_params, train_X, relu_out_train, test_X, relu_out_test)

    # featureKD_model = train_KD(hyper_params, train_X, train_Y, test_X, test_Y)

    # Streaming Label Mapping
    print('\n****************** Streaming Label Mapping ******************\n')
    hyper_params.label_mapping_hidden1 = 200
    hyper_params.label_mapping_hidden2 = 0
    hyper_params.loss = 'correlation_aware'  # label correlation-aware loss

    rest_iterations = train_Y_rest.shape[1]
    for i in range(1,rest_iterations):
        print(f"New Labels number {i} from {rest_iterations}")
        train_Y_new = train_Y_rest[:, :i+1]
        test_Y_new = test_Y_rest[:, :i+1]
        train_Y_new_tensor = torch.from_numpy(train_Y_new).float()
        hyper_params.M_new = train_Y_new.shape[1]
        hyper_params.label_mapping_output_dim = train_Y_new.shape[1]
        hyper_params.label_representation_output_dim = train_Y_new.shape[1]
        print('apply label mapping')
        # mapping_model = train_S_label_mapping(hyper_params, 0.5 * train_Y + 0.5 * soft_train_Y, train_Y_new, soft_test_Y, test_Y_new)
        # model_old = torch.load('models/{}mapping'.format(i+2))
        # torch.save(mapping_model, f'models/{i+1}mapping-pep-{split}-upd')  
        mapping_model = torch.load(
            f'models/{i+1}mapping-pep-{split}-upd')
        mapping_train_Y_new = predict(mapping_model, 0.1 * soft_train_Y + 0.9 * train_Y)
        mapping_model.eval()
        mapping_test_Y_new = predict(mapping_model,  soft_test_Y)

        # Senior Student
        mapping_train_Y_new_tensor = torch.from_numpy(mapping_train_Y_new).float()
        train_data_DSLL = CustomDataset(train_X_tensor, mapping_train_Y_new_tensor, train_Y_new_tensor)
        hyper_params.classifier_hidden1 = 200
        hyper_params.classifier_hidden2 = 100
        hyper_params.classifier_dropout = 0.1
        hyper_params.classifier_L2 = 1e-08
        hyper_params.batchNorm = False
        hyper_params.changeloss = False
        hyper_params.loss = 'correlation_aware'   # correlation_aware  correlation_entropy  entropy

        # 5, 10, 15, 20
        for batch_size in [10]:
            hyper_params.batch_size = batch_size
            # hyper_params.classifier_epoch = int(40 + 1 * hyper_params.batch_size)
            train_DSLL_loader = DataLoader(dataset=train_data_DSLL,
                                        batch_size=hyper_params.batch_size,
                                        shuffle=True,
                                        num_workers=5
                                        )
            hyper_params.label_representation_hidden1 = 200
            train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X,
                            mapping_test_Y_new, test_Y_new, train_DSLL_loader)
        break

