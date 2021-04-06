# -*- coding:utf-8 -*-

# Deep Streaming Label Learning
# Pepijn Sibbes adapted
from collections import defaultdict
import random
import sklearn.metrics as metrics
from model import _classifier, _classifier2, \
    KnowledgeDistillation,  _classifierBatchNorm,    IntegratedDSLL, LossPredictionMod, MarginRankingLoss_learning_loss, _S_label_mapping
# _BP_ML,_DNN,IntegratedModel,_label_representation, _S_label_mapping, _S_label_mapping2,

from helpers import predictor_accuracy, precision_at_ks, predict, predict_integrated, \
    print_predict, LayerActivations, modify_state_dict

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

class CustomDataset(Dataset):
    def __init__(self, x_tensor,y_mapping_tensor, y_tensor):
        self.x = x_tensor
        self.y_mapping = y_mapping_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y_mapping[index], self.y[index])

    def __len__(self):
        return len(self.x)

def train_new(model, train_X, train_Y):
    print("Start training model with old labels")
    train_X_tensor = torch.from_numpy(train_X).float()
    train_Y_tensor = torch.from_numpy(train_Y).float()
    train_data = TensorDataset(train_X_tensor, train_Y_tensor)
    train_loader = DataLoader(dataset=train_data,
                                        batch_size=8,
                                        shuffle=True,
                                        num_workers=5
                                        )
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-05)
    model.train()
    amount = 1
    epochs = 3
    for _ in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            amount+=1
    print(f"Done training old labels {amount*8} examples trained")
    return model


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step

def make_eval_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.eval()
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step


def observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                        mapping_test_Y_new, test_Y_new):
    print('[%d/%d]Loss: %.3f' % (
        hyper_params.currentEpoch + 1, hyper_params.classifier_epoch, np.mean(training_losses)))
    if ((((hyper_params.currentEpoch + 1) % 10) == 0) | ((hyper_params.currentEpoch + 1)
                                                         == hyper_params.classifier_epoch)):
        # print('train performance')
        pred_Y_train = predict_integrated(classifier, train_X, mapping_train_Y_new)
        # pred_Y_train = pred_Y_train.round()
        # print("ashdsh")
        # print(train_Y_new)
        # exit()
        _ = print_predict(train_Y_new, pred_Y_train, hyper_params)
        # print("double?")

    # if (((hyper_params.currentEpoch + 1) % 5 == 0) | (hyper_params.currentEpoch < 10)):
    print('test performance')
    pred_Y = predict_integrated(classifier, test_X, mapping_test_Y_new)
    # pred_Y = pred_Y.round()
    measurements = print_predict(test_Y_new, pred_Y, hyper_params)
    return measurements
    # return None


def observe_train(hyper_params, classifier, training_losses, train_X, train_Y, test_X, test_Y):
    print('[%d/%d]Loss: %.3f' % (
        hyper_params.currentEpoch + 1, hyper_params.classifier_epoch, np.mean(training_losses)))
    if ((((hyper_params.currentEpoch + 1) % 10) == 0) | ((hyper_params.currentEpoch + 1)
                                                         == hyper_params.classifier_epoch)):
        print('train performance')
        pred_Y_train = predict(classifier, train_X)
        _ = print_predict(train_Y, pred_Y_train, hyper_params)

    if (((hyper_params.currentEpoch + 1) % 5 == 0) | (hyper_params.currentEpoch < 10)):
        print('test performance')
        pred_Y = predict(classifier, test_X)
        measurements = print_predict(test_Y, pred_Y, hyper_params)

def train_KD(hyper_params, train_X, train_Y, test_X, test_Y):
    print(f"train_KD\ninput dim: {hyper_params.KD_input_dim} output dim: {hyper_params.KD_output_dim}")
    hyper_params.KD_input_dim = train_X.shape[1]
    hyper_params.KD_output_dim = train_Y.shape[1]
    classifier = KnowledgeDistillation(hyper_params)
    if torch.cuda.is_available():
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    criterion = nn.MSELoss()

    for epoch in range(hyper_params.KD_epoch):
        losses = []
        for i, sample in enumerate(train_X):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1)
            classifier.train()
            optimizer.zero_grad()
            output = classifier(inputv)
            # print(output,labelsv)
            # print(output.shape,labelsv.shape)
            loss = criterion(output, labelsv)

            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
        print('[%d/%d]Distillation Loss: %.3f' % (epoch + 1, hyper_params.KD_epoch, np.mean(losses)))
    print('complete the training')
    return classifier


def train_integrated_model(hyper_params, KD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new,
                                      test_X,  soft_test_Y, mapping_test_Y_new, test_Y_new):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y_new.shape[1]
    hyper_params.classifier_hidden1 = KD_model.state_dict()['W_m.0.weight'].shape[0]
    hyper_params.KD_input_dim = train_X.shape[1]
    hyper_params.kD_output_dim = hyper_params.classifier_hidden1

    classifier_W_m = KD_model
    classifier_W_m_dict = classifier_W_m.state_dict()
    if torch.cuda.is_available():
        integrated_model = IntegratedModel(hyper_params).cuda()
    else:
        integrated_model = IntegratedModel(hyper_params)

    integrated_model_dict = integrated_model.state_dict()

    classifier_W_m_dict = {k: v for k, v in classifier_W_m_dict.items() if k in integrated_model_dict}
    integrated_model_dict.update(classifier_W_m_dict)
    integrated_model.load_state_dict(integrated_model_dict, strict=False)

    # for param in integrated_model.parameters():
    #     param.requires_grad = False
    # mapping_model = torch.load('model/bestModel/10.31experiment/mapping_epoch6_64-00.5soft_0.5hard')   ,'lr': 0.0001

    # optimizer = torch.optim.Adam([
    #     {'params':integrated_model.W_m.parameters(), 'lr': 0.001},
    #     {'params':integrated_model.representation.parameters()},
    #     {'params':integrated_model.mapping_W.parameters()},
    # ], weight_decay=hyper_params.classifier_L2)

    optimizer = optim.Adam(integrated_model.parameters(), weight_decay=hyper_params.classifier_L2)

    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(hyper_params.classifier_epoch):
        hyper_params.currentEpoch = epoch
        losses = []
        for i, sample in enumerate(train_X):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1).cuda()
                mapping_y_new = Variable(torch.FloatTensor(mapping_train_Y_new[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1)
                mapping_y_new = Variable(torch.FloatTensor(mapping_train_Y_new[i])).view(1, -1)

            integrated_model.train()
            optimizer.zero_grad()
            output = integrated_model(inputv, mapping_y_new)
            loss = criterion(output, labelsv) + label_correlation_loss2(output, labelsv)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
    print('complete the training')
    return integrated_model


def train_classifier(hyper_params, train_X, train_Y, test_X, test_Y):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]

    if hyper_params.classifier_hidden2 == 0:
        classifier = _classifier(hyper_params)
    else:
        classifier = _classifier2(hyper_params)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = nn.DataParallel(classifier, device_ids=[0, 1])

    optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(hyper_params.classifier_epoch):
        losses = []
        classifier.train()

        for i, sample in enumerate(train_X):

            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1)

            optimizer.zero_grad()
            output = classifier(inputv)
            loss = criterion(output, labelsv)

            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())

    print('complete the training')
    return classifier

def lossAdd(x, y):
    loss1 = nn.MultiLabelSoftMarginLoss()
    loss = loss1(x, y) + 0.5 * label_correlation_DIYloss(x, y)
    return loss

def lossAddcorrelation(x, y):
    loss1 = nn.MultiLabelSoftMarginLoss()
    loss = loss1(x, y) + label_correlation_loss2(x, y)
    return loss

def train_classifier_batch(hyper_params, train_X, train_Y, test_X, test_Y, train_loader):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]
    # hyper_params.model_name = 'classifier'
    if hyper_params.batchNorm:
        hyper_params.model_name = 'classifier-BatchNorm'

    if hyper_params.classifier_hidden2 == 0:
        classifier = _classifier(hyper_params)
    else:
        classifier = _classifier2(hyper_params)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
    optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    # optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},
    #                          {'params': w2, 'lr': 0.001}])

    if hyper_params.loss == 'entropy':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif hyper_params.loss == 'correlation':
        criterion = label_correlation_loss2
    elif hyper_params.loss == 'correlation_entropy':
        criterion = lossAddcorrelation
    elif hyper_params.loss == 'DIY':
        criterion = DIYloss()
    elif hyper_params.loss == 'DIY_entropy':
        criterion = lossAdd
    else:
        print('please choose loss function (CrossEntropy is default)')
        criterion = nn.MultiLabelSoftMarginLoss()

    train_step = make_train_step(classifier, criterion, optimizer)
    eval_step = make_eval_step(classifier, criterion, optimizer)

    training_losses = []
    # for each epoch
    for epoch in range(hyper_params.classifier_epoch):
        batch_losses = []
        hyper_params.currentEpoch = epoch

        if ((epoch+1) % 20 == 0) & hyper_params.changeloss:
            losses = []
            classifier.train()

            for i, sample in enumerate(train_X):
                if (i+1) % 10 == 0:

                    if torch.cuda.is_available():
                        inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                        labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1).cuda()
                    else:
                        inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                        labelsv = Variable(torch.FloatTensor(train_Y[i])).view(1, -1)

                    output = classifier(inputv)
                    loss = criterion(output, labelsv) + label_correlation_loss2(output, labelsv)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data.mean().item())

            observe_train(hyper_params, classifier, losses, train_X, train_Y, test_X, test_Y)
            print('\nchange loss:', np.mean(losses))

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(hyper_params.device)
            y_batch = y_batch.to(hyper_params.device)

            loss = train_step(x_batch, y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        observe_train(hyper_params, classifier, training_losses, train_X, train_Y, test_X, test_Y)

    print('complete the training')
    return classifier


def make_train_DSLL(model, loss_fn, optimizer):
    def train_step_DSLL(x, y_mapping, y):
        # Sets model to TRAIN mode
        model.train()
        optimizer.zero_grad()
        # Makes predictions
        yhat, kd_mid, trans_mid, ss_mid = model(x,y_mapping)
        loss = loss_fn(yhat, y)
        # Computes gradients
        loss.mean().backward()
        # loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        
        return loss, kd_mid, trans_mid, ss_mid

    # Returns the function that will be called inside the train loop
    return train_step_DSLL

def make_eval_DSLL(model, loss_fn, optimizer):
    def eval_step_DSLL(x, y_mapping, y):
        # Sets model to EVAL mode
        model.eval()
        optimizer.zero_grad()
        # Makes predictions
        yhat, kd_mid, trans_mid, ss_mid = model(x,y_mapping)
        loss = loss_fn(yhat, y)        

        return loss, kd_mid, trans_mid, ss_mid

    # Returns the function that will be called inside the train loop
    return eval_step_DSLL

################################################################## DSLL ###################################################################
def train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X, mapping_test_Y_new, test_Y_new, train_loader, use_al):
    
    # if use_al is True then there is no train loader, then it is a combination of the seed and pool which each consist out of three items

    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]
    device = hyper_params.device
    hyper_params.model_name = 'DSLL'

    # create new classifier
    classifier = IntegratedDSLL(hyper_params) 

    # copy weight information from KnowledgeDistillation 1st layer to IntegratedDSLL first layer
    classifier_W_m = featureKD_model
    classifier_W_m_dict = classifier_W_m.state_dict()
    classifier_dict = classifier.state_dict()
    classifier_W_m_dict = {k: v for k, v in classifier_W_m_dict.items() if k in classifier_dict}
    classifier_dict.update(classifier_W_m_dict)

    classifier.load_state_dict(classifier_dict, strict=False)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
    # optimizer = optim.RMSprop(classifier.parameters())
    optimizer = torch.optim.Adam([
        {'params':classifier.W_m.parameters()},   # , 'lr': 0.0001},
        {'params':classifier.seniorStudent.parameters()},
        {'params':classifier.transformation.parameters()},
    ], weight_decay=hyper_params.classifier_L2, lr = 0.001)

    if hyper_params.loss == 'entropy':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif hyper_params.loss == 'correlation':
        criterion = label_correlation_loss2
    elif hyper_params.loss == 'correlation_entropy':
        criterion = lossAddcorrelation
    elif hyper_params.loss == 'DIY':
        criterion = DIYloss()
    elif hyper_params.loss == 'DIY_entropy':
        criterion = lossAdd
    else:
        #print('please choose loss function (MultiLabelSoftMarginLoss is default)')
        # criterion = nn.MultiLabelSoftMarginLoss(reduction ='none') 
        criterion = AsymmetricLoss(reduce=False)

    # train_step = make_train_DSLL(classifier, criterion, optimizer)
    eval_step = make_eval_DSLL(classifier, criterion, optimizer)

    # lp_criterions = {"ndcg": approxNDCGLoss(), "lambda":LambdaLoss(), "listnet": ListNetLoss(), "listMLE": ListMLELoss(), \
    #                               "RMSE":RMSELoss(), "rank":RankLoss(),"mapranking":MapRankingLoss(),"spearman":SpearmanLoss()}
    lp_criterion = MarginRankingLoss_learning_loss()
    
    
    classifier_lpm = LossPredictionMod(hyper_params)
    # optimizer2 = optim.Adam(classifier_lpm.parameters(), weight_decay=hyper_params.classifier_L2)
    optimizer2 = torch.optim.Adam([
            {'params':classifier_lpm.Fc1.parameters()},   # , 'lr': 0.0001},
            {'params':classifier_lpm.Fc2.parameters()},
            {'params':classifier_lpm.Fc3.parameters()},
            {'params':classifier_lpm.fc_concat.parameters()},
        ], weight_decay=hyper_params.classifier_L2, lr=0.001)

     
    if torch.cuda.is_available():
        classifier_lpm = classifier_lpm.cuda()
    # if ac:
        activation = {}
        def get_activation(name):
            # print("INN")
            def hook(model, input, output):
                activation[name] = output.detach()
                print("INN")
                print(output.detach())
            return hook


    training_losses, full_losses, full_measurements = [], [], []

    # for each epoch
    x_axis, ndcg_saved = [], []
    for epoch in range(hyper_params.classifier_epoch):
        batch_losses = []
        hyper_params.currentEpoch = epoch

        for x_batch, y_mapping, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_mapping = y_mapping.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.shape[0]

            # loss, kd_mid, trans_mid, ss_mid = train_step(x_batch, y_mapping, y_batch)
            
            classifier.train()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            yhat, kd_mid, trans_mid, ss_mid = classifier(x_batch,y_mapping)
            loss = criterion(yhat, y_batch)
            loss.mean().backward()
            optimizer.step()
            # training_losses.append(loss.mean().detach())
            # loss_predicted_loss = classifier_lpm(kd_mid, trans_mid, ss_mid)
            # if epoch <= 1:
            #     loss2 = lp_criterion(loss_predicted_loss, loss.unsqueeze(1))
            #     calc_loss = loss.mean() + loss2
            #     calc_loss.backward()
            #     optimizer2.step()
            #     optimizer.step()
            # else:
            #     calc_loss = loss.mean()
            #     calc_loss.backward()
            #     optimizer.step()
            

            # print(loss)
            # exit()
            batch_losses.append(loss.mean().item()) #.mean()

            # print(kd_mid, trans_mid, ss_mid)
            # copy weights from IntegratedDSLL model to loss prediction model
            # classifier.eval()
            if epoch <= 10 and len(batch_losses) % 5 == 0:
                measurements = observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                            mapping_test_Y_new, test_Y_new)
                if measurements != None:
                    full_measurements.append(measurements)

            # how many epochs does the loss module train
            if epoch <= 1:
                optimizer.zero_grad()
                optimizer2.zero_grad()
                try:
                    # Makes predictions detach old loss function (only update over loss prediction module)
                    kd_mid, trans_mid, ss_mid = kd_mid.detach(), trans_mid.detach(), ss_mid.detach()
                    predicted_loss = classifier_lpm(kd_mid, trans_mid, ss_mid)

                    loss2 = lp_criterion(predicted_loss, loss.unsqueeze(1).detach())
                    loss3 = loss.mean().detach() + loss2

                    # Computes gradients and updates model
                    loss3.backward()
                    optimizer2.step()
                except:
                    print("WEIRD ERROR")
                optimizer2.zero_grad()
                # if len(batch_losses) % 1 == 0:
                    # print(predicted_loss,loss.unsqueeze(1))
                    # print(f"loss prediction loss: {loss3}")
                    
                    #### TRAIN NDCG #####
                    # true loss number is number in row (first is highest)
                    # ndcg_true = np.asarray(loss.unsqueeze(1).cpu().detach().numpy())
                    # ndcg_seq = sorted(ndcg_true)
                    # ndcg_index = np.asarray([ndcg_seq.index(v) for v in ndcg_true])[..., np.newaxis]

                    # compare rank with score higher score is higher confidence so needs to match true loss rank
                    # ndcg_score =  np.asarray(predicted_loss.cpu().detach().numpy())

                    # right size for the ndcg is (1,batch_size)
                    # ndcg_index.resize(1,batch_size)
                    # ndcg_score.resize(1,batch_size)

                    # ndcg at half of batch size 
                    # batch_ndcg = metrics.ndcg_score(ndcg_index,ndcg_score, k=int(batch_size/2))
                    # TRAIN NDCG SAVED MOVED FOR TEST
                    # ndcg_saved.append(batch_ndcg)
                    
                    # print(f"batch NDCG real score: {batch_ndcg}")

                    ##### TEST NDCG #####
                    # classifier.eval()
                    # test_loss ,kd_mid_test, trans_mid_test, ss_mid_test = eval_step( torch.from_numpy(test_X).float().to(device),\
                    #                             torch.from_numpy(mapping_test_Y_new).float().to(device),\
                    #                             torch.from_numpy(test_Y_new).float().to(device))
                    # kd_mid_test, trans_mid_test, ss_mid_test = kd_mid_test.detach(), trans_mid_test.detach(), ss_mid_test.detach()
                    # predicted_loss_test = classifier_lpm(kd_mid_test, trans_mid_test, ss_mid_test)
                    # # print(f"Test loss size: {test_loss.shape[0]}, with mean of {test_loss.mean()}")
                    # ndcg_true = np.asarray(test_loss.unsqueeze(1).cpu().detach().numpy())
                    # ndcg_seq = sorted(ndcg_true)
                    # ndcg_index = np.asarray([ndcg_seq.index(v) for v in ndcg_true])[..., np.newaxis]

                    # # compare rank with score higher score is higher confidence so needs to match true loss rank
                    # ndcg_score =  np.asarray(predicted_loss_test.cpu().detach().numpy())

                    # # right size for the ndcg is (1,batch_size)
                    # ndcg_index.resize(1,test_loss.shape[0])
                    # ndcg_score.resize(1,test_loss.shape[0])

                    # # ndcg at 10 percent of test size
                    # test_ndcg = metrics.ndcg_score(ndcg_index,ndcg_score, k=int(test_loss.shape[0]*0.1))
                    # ndcg_saved.append(test_ndcg)
                    # print(f"Test NDCG: {test_ndcg}")
                    # classifier.train()
                
        full_losses.append(batch_losses)
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        # measurements2 = observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
        #                    mapping_test_Y_new, test_Y_new)
        if measurements != None:
            print(measurements)
            full_measurements.append(measurements)

    # print(full_losses)
    # print losses figure
    # [F1_Micro, F1_Macro, AUC_micro, AUC_macro]
    F1_Micro = np.hstack(np.array(full_measurements)[:,0])
    F1_Macro = np.hstack(np.array(full_measurements)[:,1])
    AUC_micro = np.hstack(np.array(full_measurements)[:,2])
    AUC_macro = np.hstack(np.array(full_measurements)[:,3])
    # print(f"{F1_Micro}\n{F1_Macro}\n{AUC_micro}\n{AUC_macro}")
    x_axis = [i for i in range(len(F1_Micro))]
    import matplotlib.pyplot as plt
    plt.plot(x_axis, F1_Micro, label='F1_Micro')
    plt.plot(x_axis, F1_Macro, label='F1_Macro')
    plt.plot(x_axis, AUC_micro, label='AUC_micro')
    plt.plot(x_axis, AUC_macro, label='AUC_macro')
    plt.xlabel('Instances', fontsize=18)
    plt.ylabel('Values', fontsize=16)
    # plt.ylim(0, 1)
    # from scipy.stats import linregress

    # slope, intercept, r_value, p_value, std_err = linregress(x_axis, F1_Micro)
    # print(f"Slope is: {slope*1000}")
    # print(f"Mean NDCG: {np.mean(F1_Micro)}")
    # plt.plot(np.unique(x_axis), np.poly1d(np.polyfit(x_axis, F1_Micro, 1))(np.unique(x_axis)))
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.show()
    # exit()

    print('complete the training')
    return classifier

def set_random_seed(seednr):
    random.seed(seednr)
    torch.manual_seed(seednr)
    torch.cuda.manual_seed(seednr)
    np.random.seed(seednr)
    random.seed(seednr)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

################################################################## AL DSLL ##################################################################
def AL_train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X, mapping_test_Y_new, test_Y_new, train_loader, use_al, seednr):
    
    # if use_al is True then there is no train loader, then it is a combination of the seed and pool which each consist out of three items

    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]
    device = hyper_params.device
    hyper_params.model_name = 'DSLL'

    from collections import defaultdict
    training_losses, full_losses, full_measurements = [], [], defaultdict(list)

    # base of active learning train with the seed then add from pool the best examples and train more
    xseed_save, y_mappingseed_save, yseed_save, xpool_save, y_mappingpool_save, ypool_save = train_loader
    batch_size = len(xseed_save)

    # how many times (with different seed) do you want to run the random active learner (for smooth graph) leave at 5
    # random_amount = 5
    
    # svm and rf callable classes for the forward passes
    svm_al_learner = ActiveSVMLearner()
    forest_al_learner = ActiveForestLearner()
    worstcasesvm_al_learner = ActiveForestLearnerWorstCase()

    # which of the loss prediction module selection procedures you want to use
    # choose{original, kmeans, distance}
    lpm_selection = "kmeans" #"kmeans"
    
    # for now altype can be worstcase, random, forest, svm, or lpm
    # set the range for which active learning methods you want to see
    altypes = ['worstcase', 'svm', 'random',  'rf',  'lpm', 'lpm_pointwise', 'lpm_rankwise'][4:5]
    for altype in altypes:
        # create new classifier
        classifier = IntegratedDSLL(hyper_params) 

        # copy weight information from KnowledgeDistillation 1st layer to IntegratedDSLL first layer
        classifier_W_m = featureKD_model
        classifier_W_m_dict = classifier_W_m.state_dict()
        classifier_dict = classifier.state_dict()
        classifier_W_m_dict = {k: v for k, v in classifier_W_m_dict.items() if k in classifier_dict}
        classifier_dict.update(classifier_W_m_dict)

        classifier.load_state_dict(classifier_dict, strict=False)

        if torch.cuda.is_available():
            classifier = classifier.cuda()

        optimizer = torch.optim.Adam([
            {'params':classifier.W_m.parameters()},   # , 'lr': 0.0001},
            {'params':classifier.seniorStudent.parameters()},
            {'params':classifier.transformation.parameters()},
        ], weight_decay=hyper_params.classifier_L2, lr = 0.001)

        # main loss function
        criterion = AsymmetricLoss(reduce=False)

        # train_step = make_train_DSLL(classifier, criterion, optimizer)
        eval_step = make_eval_DSLL(classifier, criterion, optimizer)

        lp_criterion = MarginRankingLoss_learning_loss()
        
        
        classifier_lpm = LossPredictionMod(hyper_params)
        # optimizer2 = optim.Adam(classifier_lpm.parameters(), weight_decay=hyper_params.classifier_L2)
        optimizer2 = torch.optim.Adam([
                {'params':classifier_lpm.Fc1.parameters()},   # , 'lr': 0.0001},
                {'params':classifier_lpm.Fc2.parameters()},
                {'params':classifier_lpm.Fc3.parameters()},
                {'params':classifier_lpm.fc_concat.parameters()},
            ], weight_decay=hyper_params.classifier_L2, lr=0.001)
        
        classifier_lpm_old = LossPredictionMod(hyper_params)
        # optimizer2 = optim.Adam(classifier_lpm.parameters(), weight_decay=hyper_params.classifier_L2)
        optimizer3 = torch.optim.Adam([
                {'params':classifier_lpm_old.Fc1.parameters()},   # , 'lr': 0.0001},
                {'params':classifier_lpm_old.Fc2.parameters()},
                {'params':classifier_lpm_old.Fc3.parameters()},
                {'params':classifier_lpm_old.fc_concat.parameters()},
            ], weight_decay=hyper_params.classifier_L2, lr=0.001)

        
        if torch.cuda.is_available():
            classifier_lpm = classifier_lpm.cuda()
            classifier_lpm_old = classifier_lpm_old.cuda()

        # reload the original (saved) values
        xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool = xseed_save, y_mappingseed_save, yseed_save, xpool_save, y_mappingpool_save, ypool_save
        
        # what is the max budget possible
        max_budget = int(len(xpool)/batch_size)
        # how big is the active learning budget
        # make the budget the same or half as the max
        budget = int(max_budget//2) #int(max_budget/2)
        # for each epoch
        x_axis, ndcg_saved = [], []

        # do many tests for a smoother line TODO V0.3 make work for other active learners
        # if altype == 'random':
        random_amount = 3 if altype == "random" else 2
        if altype == "random":
            full_measurements[altype] = do_many_active(altype, budget, device, xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, hyper_params, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                                mapping_test_Y_new, test_Y_new, batch_size, random_amount, featureKD_model)
            continue

        # kmeans initialized with the original xpool, the df2 contains its labels and value get deleted when chosen for seed
        if lpm_selection == "kmeans":
            # we can incease/decrease the batch_size multiplier between 2-4
            # best results: multiplier:3, ncomponents:4
            batch_size_multiplier = 3
            kmeans = KMeans(
                init="random",
                n_clusters=batch_size*batch_size_multiplier,
                n_init=10,
                max_iter=300,
                random_state=42
            )
            # try with and without standard scaler
            # try with dimensionality reduction?
            from sklearn.decomposition import PCA
            pca = PCA(n_components=4)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(xpool)
            principalComponents = pca.fit_transform(scaled_features)
            
            # kmeans.fit(scaled_features)
            kmeans.fit(principalComponents)
            df2 = pd.DataFrame(kmeans.labels_, columns = ['cluster'])


        set_random_seed(seednr)


        # how many epochs need to happen
        hyper_params.classifier_epoch = 1
        for epoch in range(hyper_params.classifier_epoch):
            batch_losses = []
            hyper_params.currentEpoch = epoch
            
            # for x_batch, y_mapping, y_batch in train_loader:
            for bud in range(budget):
                # just take the last batch_size items to train, these are selected out of the pool to train on
                x_batch, y_mapping, y_batch = xseed[-batch_size:], y_mappingseed[-batch_size:], yseed[-batch_size:]
                x_batch = x_batch.to(device)
                y_mapping = y_mapping.to(device)
                y_batch = y_batch.to(device)
                batch_size = x_batch.shape[0]
                
                # train with the seed and later the pool
                classifier.train()
                optimizer.zero_grad()
                optimizer2.zero_grad()
                yhat, kd_mid, trans_mid, ss_mid = classifier(x_batch,y_mapping)
                loss = criterion(yhat, y_batch)
                loss.mean().backward()
                optimizer.step()           

                batch_losses.append(loss.mean().item()) #.mean()

                # obtain measurements
                if epoch <= 10 : # and len(batch_losses) % 1 == 0
                    measurements = observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                                mapping_test_Y_new, test_Y_new)
                    if measurements != None:
                        full_measurements[altype].append(measurements)

                ############## ACTIVE LEARNING PART ##############
                # use one of the active learning selection procedures
                if altype == "random":
                    chosen_indices = active_random(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                elif altype == "worstcase":
                    # chosen_indices = active_worstcase(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                    chosen_indices = worstcasesvm_al_learner.forward(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                elif altype == "svm":
                    chosen_indices = svm_al_learner.forward(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                elif altype == "rf":
                    chosen_indices = forest_al_learner.forward(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                elif altype == "lpm":
                    # Loss learning module

                    # train the loss prediction module with the seed
                    if epoch <= 1:
                        optimizer.zero_grad()
                        optimizer2.zero_grad()
                        optimizer3.zero_grad()
                        try:
                            # Makes predictions detach old loss function (only update over loss prediction module)
                            kd_mid, trans_mid, ss_mid = kd_mid.detach(), trans_mid.detach(), ss_mid.detach()
                            predicted_loss = classifier_lpm(kd_mid, trans_mid, ss_mid)

                            loss2 = lp_criterion(predicted_loss, loss.unsqueeze(1).detach())
                            loss3 = loss.mean().detach() + loss2

                            # Computes gradients and updates model
                            loss3.backward()
                            optimizer2.step()
                        except:
                            print("WEIRD ERROR")
                        optimizer2.zero_grad()
                        optimizer3.zero_grad()

                    # predict best items to select in the pool
                    classifier.eval()
                    # get the model output of the xpools (no training)
                    _, kd_mid_test, trans_mid_test, ss_mid_test = classifier(xpool.to(device),y_mappingpool.to(device))
                    kd_mid_test, trans_mid_test, ss_mid_test = kd_mid_test.detach(), trans_mid_test.detach(), ss_mid_test.detach()
                    predicted_loss_test = classifier_lpm(kd_mid_test, trans_mid_test, ss_mid_test)
                    # # print(f"Test loss size: {test_loss.shape[0]}, with mean of {test_loss.mean()}")
                    # true_ranking = np.asarray(yhat_test.unsqueeze(1).cpu().detach().numpy())
                    predicted_losses_array =  np.asarray(predicted_loss_test.cpu().detach().numpy())

                    if lpm_selection == "original":
                        # the original loss prediction module
                        # get the top 10 highest loss indices, 10 is here batch_size
                        top_10_loss_indices = np.argpartition(predicted_losses_array,-batch_size, axis=0)[-batch_size:]
                        # these are in a list of a list so unpack it
                        chosen_indices = [i[0] for i in top_10_loss_indices]
                    elif lpm_selection == "kmeans":
                        
                        # make a dataframe with two columns, the losses and cluster number of the items in the pool and sort them
                        # the index number is kept so we have the index of the highest loss items in the top
                        # df1, df2 = pd.DataFrame(predicted_losses_array, columns = ['losses']), pd.DataFrame(kmeans.labels_, columns = ['cluster'])
                        # df2 are the labels from the pool
                        df1= pd.DataFrame(predicted_losses_array, columns = ['losses'])
                        df3 = pd.concat([df1, df2], axis=1)
                        df4 = df3.sort_values('losses',ascending=False)

                        chosen_indices, chosen_clusters = [], []
                        # choose the items if the cluster is not already represented (chosen)
                        for i in df4.iterrows():
                            # stop condition if we have reached the batch size
                            if len(chosen_indices) == batch_size:
                                break
                            # add to chosen indices if it is not in chosen_clusters (inherently from high loss to low)
                            if int(i[1][1]) not in chosen_clusters:
                                chosen_clusters.append(int(i[1][1]))
                                chosen_indices.append(int(i[0]))
                                # make sure the index is the same in df2 and df4
                                assert int(df4.loc[int(i[0])].cluster) == int(df2.loc[int(i[0])].cluster)
                                # drop the chosen index from the pool
                                df2.drop(int(i[0]))
                        # top_10_loss_indices = np.argpartition(predicted_losses_array,-batch_size, axis=0)[-batch_size:]
                        # these are in a list of a list so unpack it
                        # chosen_indices2 = [i[0] for i in top_10_loss_indices]
                        # chosen_indices.sort()
                        # chosen_indices2.sort()
                        # print(chosen_indices)
                        # print(chosen_indices2)
                        # print(set(chosen_indices)- set(chosen_indices2))

                        # choose different clusters just as now or closer to center clusters
                        # print(chosen_indices)
                        # print(chosen_clusters)

                        # print(df4)
                        # print(df4['cluster'][:20])
                        # print(kmeans.cluster_centers_.shape)
                        # print(kmeans.cluster_centers_[5].shape)
                        # exit()
                        # if bud >= int(budget//2):
                        #     lpm_selection = "original"
                        
                    elif lpm_selection == "distance":
                        pass
                    else:
                        print("no lpm selection measure given")
                        pass
                # original thing
                elif altype == "lpm_rankwise":
                    # Loss learning module

                    # train the loss prediction module with the seed
                    if epoch <= 1:
                        optimizer.zero_grad()
                        optimizer2.zero_grad()
                        optimizer3.zero_grad()
                        # try:
                        # Makes predictions detach old loss function (only update over loss prediction module)
                        
                        kd_mid, trans_mid, ss_mid = kd_mid.detach(), trans_mid.detach(), ss_mid.detach()
                        
                        predicted_loss = classifier_lpm_old(kd_mid, trans_mid, ss_mid)
                        
                        loss2 = lp_criterion(predicted_loss, loss.unsqueeze(1).detach())
                        loss3 = loss.mean().detach() + loss2

                        # Computes gradients and updates model
                        loss3.backward()
                        optimizer3.step()
                        # except:
                        #     print("WEIRD ERROR")
                        optimizer3.zero_grad()

                    # predict best items to select in the pool
                    classifier.eval()
                    # get the model output of the xpools (no training)
                    _, kd_mid_test, trans_mid_test, ss_mid_test = classifier(xpool.to(device),y_mappingpool.to(device))
                    kd_mid_test, trans_mid_test, ss_mid_test = kd_mid_test.detach(), trans_mid_test.detach(), ss_mid_test.detach()
                    predicted_loss_test = classifier_lpm(kd_mid_test, trans_mid_test, ss_mid_test)

                    # true_ranking = np.asarray(yhat_test.unsqueeze(1).cpu().detach().numpy())
                    predicted_losses_array =  np.asarray(predicted_loss_test.cpu().detach().numpy())

                    # get the top 10 highest loss indices
                    top_10_loss_indices = np.argpartition(predicted_losses_array,-batch_size, axis=0)[-batch_size:]
                    # these are in a list of a list so unpack it
                    chosen_indices = [i[0] for i in top_10_loss_indices]

                # add selected items to the seed
                xseed = torch.cat((xseed,xpool[chosen_indices]),0)
                y_mappingseed = torch.cat((y_mappingseed,y_mappingpool[chosen_indices]),0)
                yseed = torch.cat((yseed,ypool[chosen_indices]),0)

                # calculate which items need to be deleted from the pool (the ones that are chosen==new seed)
                all_indices = np.arange(0, len(xpool))
                non_chosen_items = list(set(all_indices) - set(chosen_indices))

                # remove the seeds from the pool
                xpool = xpool[non_chosen_items]
                y_mappingpool = y_mappingpool[non_chosen_items]
                ypool = ypool[non_chosen_items]
                    
                    
            full_losses.append(batch_losses)
            training_loss = np.mean(batch_losses)
            training_losses.append(training_loss)

            if measurements != None:
                print(measurements)
                full_measurements[altype].append(measurements)
    
    # plot active learning results
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 7))
    sns.set_style("darkgrid")
    # define colours by hand (because easier to know what is happening)
    colours = [["7215CC","008000", "00A000", "00C000", "00E000"], ["2CFF72","800000", "A00000", "C00000", "E00000"], ["D6D243","000080", "0000A0", "0000C0", "0000E0"], ["F74D7E","404000", "808000", "B0B000", "F0F000"], ["3E8DFF","404000", "808000", "B0B000", "F0F000"]]
    # loop over the al methods and calculate the measures
    # print(full_measurements)
    for e, i in enumerate(altypes):
        F1_Micro = np.hstack(np.array(full_measurements[i])[:,0])
        F1_Macro = np.hstack(np.array(full_measurements[i])[:,1])
        AUC_micro = np.hstack(np.array(full_measurements[i])[:,2])
        AUC_macro = np.hstack(np.array(full_measurements[i])[:,3])
        
        # define x-axis as arange of the length of the array
        x_axis = [i for i in range(len(F1_Micro))]
        
        # plot the lines
        plt.plot(x_axis, F1_Micro, label=f'F1_Micro_{i}',color  = f'#{colours[e][0]}')
        # plt.plot(x_axis, F1_Macro, label=f'F1_Macro_{i}',color  = f'#{colours[e][1]}')
        # plt.plot(x_axis, AUC_micro, label=f'AUC_micro_{i}',color= f'#{colours[e][2]}')
        # plt.plot(x_axis, AUC_macro, label=f'AUC_macro_{i}',color= f'#{colours[e][3]}')


    plt.xlabel('Instances', fontsize=18)
    plt.ylabel('Values', fontsize=16)

    # from scipy.stats import linregress
    plt.title("Active learning with DSLL")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.show()
    # exit()

    print('complete the training')
    return classifier


def do_many_active(active_learner, budget, device, xseed_old, y_mappingseed_old, yseed_old, xpool_old, y_mappingpool_old, ypool_old, hyper_params, training_losses, train_X_old, mapping_train_Y_new_old, train_Y_new_old, test_X_old,
                                mapping_test_Y_new_old, test_Y_new_old, batch_size, max_seednr, featureKD_model, epochs =1):
    full_measurements = defaultdict(list)
    batch_losses = []
    # which selection procedure used
    lpm_selection = "kmeans"
    # max_seednr = random_amount
    startseed = 10
    for seednr in range(startseed, startseed+max_seednr):
        worstcasesvm_al_learner = ActiveForestLearnerWorstCase()
        for epoch in range(epochs):

            set_random_seed(seednr)
            # load in the old values
            xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool = xseed_old, y_mappingseed_old, yseed_old, xpool_old, y_mappingpool_old, ypool_old
            train_X, mapping_train_Y_new, train_Y_new, test_X, mapping_test_Y_new, test_Y_new = train_X_old, mapping_train_Y_new_old, train_Y_new_old, test_X_old, mapping_test_Y_new_old, test_Y_new_old
            # reinitialise the active learners with new seed
            # svm_al_learner = ActiveSVMLearner()
            # forest_al_learner = ActiveForestLearner()

            # create new classifier
            classifier = IntegratedDSLL(hyper_params) 

            # copy weight information from KnowledgeDistillation 1st layer to IntegratedDSLL first layer
            classifier_W_m = featureKD_model
            classifier_W_m_dict = classifier_W_m.state_dict()
            classifier_dict = classifier.state_dict()
            classifier_W_m_dict = {k: v for k, v in classifier_W_m_dict.items() if k in classifier_dict}
            classifier_dict.update(classifier_W_m_dict)

            classifier.load_state_dict(classifier_dict, strict=False)

            if torch.cuda.is_available():
                classifier = classifier.cuda()

            optimizer = torch.optim.Adam([
                {'params':classifier.W_m.parameters()},   # , 'lr': 0.0001},
                {'params':classifier.seniorStudent.parameters()},
                {'params':classifier.transformation.parameters()},
            ], weight_decay=hyper_params.classifier_L2, lr = 0.001)

            # main loss function
            criterion = AsymmetricLoss(reduce=False)

            # train_step = make_train_DSLL(classifier, criterion, optimizer)
            eval_step = make_eval_DSLL(classifier, criterion, optimizer)

            lp_criterion = MarginRankingLoss_learning_loss()
            
            
            classifier_lpm = LossPredictionMod(hyper_params)
            # optimizer2 = optim.Adam(classifier_lpm.parameters(), weight_decay=hyper_params.classifier_L2)
            optimizer2 = torch.optim.Adam([
                    {'params':classifier_lpm.Fc1.parameters()},   # , 'lr': 0.0001},
                    {'params':classifier_lpm.Fc2.parameters()},
                    {'params':classifier_lpm.Fc3.parameters()},
                    {'params':classifier_lpm.fc_concat.parameters()},
                ], weight_decay=hyper_params.classifier_L2, lr=0.001)

            
            if torch.cuda.is_available():
                classifier_lpm = classifier_lpm.cuda()

            # for x_batch, y_mapping, y_batch in train_loader:
            for e in range(budget):
                # just take the last batch_size items to train, these are selected out of the pool to train on
                x_batch, y_mapping, y_batch = xseed[-batch_size:], y_mappingseed[-batch_size:], yseed[-batch_size:]
                x_batch = x_batch.to(device)
                y_mapping = y_mapping.to(device)
                y_batch = y_batch.to(device)
                batch_size = x_batch.shape[0]
                
                # train with the seed and later the pool
                classifier.train()
                optimizer.zero_grad()
                optimizer2.zero_grad()
                yhat, kd_mid, trans_mid, ss_mid = classifier(x_batch,y_mapping)
                loss = criterion(yhat, y_batch)
                loss.mean().backward()
                optimizer.step()           

                batch_losses.append(loss.mean().item()) #.mean()

                # obtain measurements
                # measurements is list of 4 items
                measurements = observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                                    mapping_test_Y_new, test_Y_new)
                
                # specialcase for first item of the new seed, the first measurements list needs to be appended to the full lists
                # the following items will just be added to this list. Numpy can actually add 2 lists so this is used
                # for measurements we divide by the max number
                # one list per seed, so budget amount of lists to which we add the other lists
                if measurements != None and seednr == startseed:
                    measurements = np.array(measurements)/max_seednr
                    full_measurements[active_learner].append(measurements.tolist())
                # numpy can add values in lists, python just makes bigger listts
                elif measurements != None:
                    pythonlist = full_measurements[active_learner][e] 
                    numpylist = np.array(pythonlist) + np.array(measurements)/max_seednr
                    full_measurements[active_learner][e] = numpylist.tolist()
                    # print(np.array(full_measurements[active_learner]).shape)
                    # print(e)
                    # print(full_measurements[active_learner][e])
                    # exit()
                



                ############## ACTIVE LEARNING PART ##############
                if active_learner == "random":
                    chosen_indices = active_random(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                elif active_learner == "worstcase":
                    # chosen_indices = active_worstcase(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                    chosen_indices = worstcasesvm_al_learner.forward(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                    # chosen_indices = np.arange(int(batch_size)).tolist()
                elif active_learner == "svm":
                    chosen_indices = svm_al_learner.forward(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                elif active_learner == "rf":
                    chosen_indices = forest_al_learner.forward(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size)
                elif active_learner == "lpm":
                    # Loss learning module

                    # train the loss prediction module with the seed
                    # if epoch <= 1:
                    optimizer.zero_grad()
                    optimizer2.zero_grad()
                    # try:
                    # Makes predictions detach old loss function (only update over loss prediction module)
                    kd_mid, trans_mid, ss_mid = kd_mid.detach(), trans_mid.detach(), ss_mid.detach()
                    predicted_loss = classifier_lpm(kd_mid, trans_mid, ss_mid)

                    loss2 = lp_criterion(predicted_loss, loss.unsqueeze(1).detach())
                    loss3 = loss.mean().detach() + loss2

                    # Computes gradients and updates model
                    loss3.backward()
                    optimizer2.step()
                    # except:
                    #     print("WEIRD ERROR")
                    optimizer2.zero_grad()

                    # predict best items to select in the pool
                    classifier.eval()
                    # get the model output of the xpools (no training)
                    _, kd_mid_test, trans_mid_test, ss_mid_test = classifier(xpool.to(device),y_mappingpool.to(device))
                    kd_mid_test, trans_mid_test, ss_mid_test = kd_mid_test.detach(), trans_mid_test.detach(), ss_mid_test.detach()
                    predicted_loss_test = classifier_lpm(kd_mid_test, trans_mid_test, ss_mid_test)
                    # # print(f"Test loss size: {test_loss.shape[0]}, with mean of {test_loss.mean()}")
                    # true_ranking = np.asarray(yhat_test.unsqueeze(1).cpu().detach().numpy())
                    predicted_losses_array =  np.asarray(predicted_loss_test.cpu().detach().numpy())
                    
                    if lpm_selection == "original":
                        # the original loss prediction module
                        # get the top 10 highest loss indices
                        top_10_loss_indices = np.argpartition(predicted_losses_array,-batch_size, axis=0)[-batch_size:]
                        # these are in a list of a list so unpack it
                        chosen_indices = [i[0] for i in top_10_loss_indices]
                    elif lpm_selection == "kmeans":
                        kmeans = KMeans(
                            init="random",
                            n_clusters=batch_size*4,
                            n_init=10,
                            max_iter=300,
                            random_state=42
                        )
                        scaler = StandardScaler()
                        scaled_features = scaler.fit_transform(xpool)
                        kmeans.fit(scaled_features)

                        
                        # kmeans selection
                        # print(predicted_losses_array.shape[0])
                        # print(kmeans.labels_.shape[0])
                        # make sure that kmeans will work
                        assert predicted_losses_array.shape[0] == kmeans.labels_.shape[0]

                        # make a dataframe with two columns, the losses and cluster number of the items in the pool and sort them
                        # the index number is kept so we have the index of the highest loss items in the top
                        df1, df2 = pd.DataFrame(predicted_losses_array, columns = ['losses']), pd.DataFrame(kmeans.labels_, columns = ['cluster'])
                        df3 = pd.concat([df1, df2], axis=1)
                        df4 = df3.sort_values('losses',
                                        ascending=False)
                        chosen_indices, chosen_clusters = [], []
                        # choose the items if the cluster is not already represented (chosen)
                        for i in df4.iterrows():
                            # stop condition
                            if len(chosen_indices) == 10:
                                break
                            # add to chosen indices if it is not chosen (inherently from high loss to low)
                            if int(i[1][1]) not in chosen_clusters:
                                chosen_clusters.append(int(i[1][1]))
                                chosen_indices.append(int(i[0]))

                        # print(chosen_indices)
                        # print(chosen_clusters)

                        # print(df4)
                        # print(df4['cluster'][:20])
                        # print(kmeans.cluster_centers_.shape)
                        # print(kmeans.cluster_centers_[5].shape)
                        # exit()
                    elif lpm_selection == "distance":
                        pass
                    else:
                        print("no lpm selection measure given")
                        pass

                # add selected items to the seed
                xseed = torch.cat((xseed,xpool[chosen_indices]),0)
                y_mappingseed = torch.cat((y_mappingseed,y_mappingpool[chosen_indices]),0)
                yseed = torch.cat((yseed,ypool[chosen_indices]),0)

                # calculate which items need to go from the pool
                all_indices = np.arange(0, len(xpool))
                non_chosen_items = list(set(all_indices) - set(chosen_indices))

                # remove the seeds from the pool
                xpool = xpool[non_chosen_items]
                y_mappingpool = y_mappingpool[non_chosen_items]
                ypool = ypool[non_chosen_items]
    
    return full_measurements[active_learner]
                
        # full_losses.append(batch_losses)
        # training_loss = np.mean(batch_losses)
        # training_losses.append(training_loss)

# just randomly select indices without further information
def active_random(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size):
    random_seed_list = np.random.randint(0, len(xpool), batch_size)
    return random_seed_list

# select indices with a stupid idea
def active_worstcase(xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size):
    # sum over first axis
    horizontal_sum_list = np.sum(xpool.numpy(),1)
    # get random number between the min an max of this list
    ran_number = np.random.randint(np.min(horizontal_sum_list), np.max(horizontal_sum_list))
    # get the index of a item closest to this number
    worst_seed = np.argmin(np.abs(np.subtract.outer(horizontal_sum_list, ran_number)),0)
    # make a list and select 10 items close to the worst seed
    max_val= len(horizontal_sum_list)
    if worst_seed + 5 < max_val and worst_seed - 5 >= 0:
        bad_metric = np.arange(worst_seed-5,worst_seed+5)
    else:
        bad_metric = np.arange(10)

    return bad_metric

from modAL.models import ActiveLearner
from modAL.multilabel import avg_score, max_score, min_confidence, avg_confidence
from modAL.uncertainty import uncertainty_sampling
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

# select indices from xpool which have some meaning for the svm
class ActiveSVMLearner:
    def __init__(self):
        # probably the xseed and yseed are in tensors and they need to be in numpy 
        # initializing the active learner
        self.learner = ActiveLearner(
            estimator=OneVsRestClassifier(SVC(probability=True, gamma='auto')),
            query_strategy=uncertainty_sampling
            # X_training=xseed.numpy(), y_training=yseed.numpy()
        )
    # teach the learner with the seed and select the best indices out of the pool
    def forward(self, xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size):
        # only take teach the items at the end of the seed --> each time new items being trained
        self.learner.teach(xseed.numpy()[-batch_size:], yseed.numpy()[-batch_size:])
        query_idx, query_instance = self.learner.query(xpool.numpy(),  n_instances=batch_size)
        return query_idx

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
def RF_regression_sum(regressor, X):
    # predict rf
    std = regressor.predict(X)
    # sum label prediction
    std = np.sum(std, axis=1)
    # take top 10 predictions for active learning
    query_idx = np.argpartition(std,-10)[-10:]
    return query_idx, X[query_idx]

def RF_regression_sum_worstcase(regressor, X):
    # predict rf
    std = regressor.predict(X)
    # sum label prediction
    std = np.sum(std, axis=1)
    lowest10 = np.argpartition(std, 10)[:10]
    query_idx = lowest10[np.argsort(std[lowest10])]
    return query_idx, X[query_idx]
    
# run like  100 times curves mostly for random
# pretrsain on known labels
# worse than random heuristic to see what happens
# classifier chain or something as other model next to DSLL
# writing training procedures in pseudocode, with yeast and real world 
# dataset

# select indices from xpool which have some meaning for the random forest
class ActiveForestLearner:
    def __init__(self):
        # probably the xseed and yseed are in tensors and they need to be in numpy 
        # initializing the active learner
        self.learner = ActiveLearner(
            estimator=RandomForestRegressor(),
            query_strategy=RF_regression_sum,
        )

    def forward(self, xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size):
        # print(xseed[-batch_size:].shape, yseed[-batch_size:].shape)
        # only take teach the new items
        self.learner.teach(xseed.numpy()[-batch_size:], yseed.numpy()[-batch_size:])
        
        query_idx, query_instance = self.learner.query(xpool.numpy()) #
        
        return query_idx

# select indices from xpool which have some meaning for the random forest
class ActiveForestLearnerWorstCase:
    def __init__(self):
        # probably the xseed and yseed are in tensors and they need to be in numpy 
        # initializing the active learner
        self.learner = ActiveLearner(
            estimator=RandomForestRegressor(),
            query_strategy=RF_regression_sum_worstcase,
        )

    def forward(self, xseed, y_mappingseed, yseed, xpool, y_mappingpool, ypool, batch_size):
        # print(xseed[-batch_size:].shape, yseed[-batch_size:].shape)
        # only take teach the new items
        self.learner.teach(xseed.numpy()[-batch_size:], yseed.numpy()[-batch_size:])
        
        query_idx, query_instance = self.learner.query(xpool.numpy()) #
        
        return query_idx



class AsymmetricLoss(nn.Module):
    def __init__(self, reduce=True, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduce = reduce

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w
        # print(-loss.sum(1))
        return -loss.sum() if self.reduce == True else -loss.sum(1)


def train_S_label_mapping(hyper_params, train_Y, train_Y_new):
    hyper_params.label_mapping_input_dim = train_Y.shape[1]
    hyper_params.label_mapping_output_dim = train_Y_new.shape[1]
    title1 = ['train_S_label_mapping', 'input_dim={}, '.format(hyper_params.label_mapping_input_dim),
              'output_dim={}, '.format(hyper_params.label_mapping_output_dim),
              'dropout rate={}, '.format(hyper_params.label_mapping_dropout),
              'hidden1={}, '.format(hyper_params.label_mapping_hidden1),
              'hidden2={}, '.format(hyper_params.label_mapping_hidden2),
              'epoch={}'.format(hyper_params.label_mapping_epoch),  'L2={}'.format(hyper_params.label_mapping_L2)
              ]
    print(f"In label mapping\n{title1}")
    if hyper_params.label_mapping_hidden2 == 0:
        S_label_mapping = _S_label_mapping(hyper_params)
    else:
        S_label_mapping = _S_label_mapping2(hyper_params)
    if torch.cuda.is_available():
        S_label_mapping = S_label_mapping.cuda()
    optimizer_S = optim.Adam(S_label_mapping.parameters(), weight_decay=hyper_params.label_mapping_L2)
    criterion_S = nn.MultiLabelSoftMarginLoss()
    for epoch in range(hyper_params.label_mapping_epoch):
        losses = []
        for i, sample in enumerate(train_Y):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_Y_new[i])).view(1, -1)

            output = S_label_mapping(inputv)
            # output = output.sigmoid().round()
            if hyper_params.loss == 'correlation_aware':
                loss = criterion_S(output, labelsv) + label_correlation_loss2(output, labelsv)
            else:
                loss = criterion_S(output, labelsv)

            optimizer_S.zero_grad()
            loss.backward()
            optimizer_S.step()
            losses.append(loss.data.mean().item())
        print('S (label mapping) [%d/%d] Loss: %.3f' % (epoch + 1, hyper_params.label_mapping_epoch, np.mean(losses)))
    print('complete the label mapping')
    return S_label_mapping


# def train_label_representation(hyper_params, train_X, mapping_soft_train_Y_new, train_Y_new, test_X, mapping_soft_test_Y_new, test_Y_new):
#     hyper_params.label_representation_input_dim = train_X.shape[1] + mapping_soft_train_Y_new.shape[1]
#     hyper_params.label_representation_output_dim = train_Y_new.shape[1]
#     label_representation = _label_representation(hyper_params)
#     if torch.cuda.is_available():
#         label_representation = label_representation.cuda()
#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         label_representation = nn.DataParallel(label_representation, device_ids=[0, 1])


#     optimizer_label_repre = optim.Adam(label_representation.parameters(),
#                                        weight_decay=hyper_params.label_representation_L2)
#     criterion_label_repre = nn.MultiLabelSoftMarginLoss()  # nn.MultiLabelSoftMarginLoss()
#     train_input = np.hstack((train_X, mapping_soft_train_Y_new))
#     train_output_true = train_Y_new
#     test_input = np.hstack((test_X, mapping_soft_test_Y_new))
#     test_output_true = test_Y_new
#     for epoch in range(hyper_params.label_representation_epoch):
#         losses = []
#         for i, sample in enumerate(train_input):
#             if torch.cuda.is_available():
#                 inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
#                 labelsv = Variable(torch.FloatTensor(train_output_true[i])).view(1, -1).cuda()
#             else:
#                 inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
#                 labelsv = Variable(torch.FloatTensor(train_output_true[i])).view(1, -1)

#             output = label_representation(inputv)
#             loss = criterion_label_repre(output, labelsv)

#             optimizer_label_repre.zero_grad()
#             loss.backward()
#             optimizer_label_repre.step()
#             losses.append(loss.data.mean().item())
#     print('complete the label representation')
#     return label_representation

class RankHardLoss(torch.nn.Module):
    """ Loss function  inspired by hard negative triplet loss, directly applied in the rank domain """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, margin=0.2, nmax=1):
        super(RankHardLoss, self).__init__()
        self.nmax = nmax
        self.margin = margin

        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

    def hc_loss(self, scores):
        rank = self.sorter(scores)

        diag = rank.diag()

        rank = rank + torch.diag(torch.ones(rank.diag().size(), device=rank.device) * 50.0)

        sorted_rank, _ = torch.sort(rank, 1, descending=False)

        hard_neg_rank = sorted_rank[:, :self.nmax]

        loss = torch.sum(torch.clamp(-hard_neg_rank + (1.0 / (scores.size(1)) + diag).view(-1, 1).expand_as(hard_neg_rank), min=0))

        return loss

    def forward(self, scores):
        """ Expect a score matrix with scores of the positive pairs are on the diagonal """
        caption_loss = self.hc_loss(scores)
        image_loss = self.hc_loss(scores.t())

        image_caption_loss = caption_loss + image_loss

        return image_caption_loss

# class RankLoss(torch.nn.Module):
#     """ Loss function  inspired by recall """
#     def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None,):
#         super(RankLoss, self).__init__()
#         self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

#     def forward(self, scores):
#         """ Expect a score matrix with scores of the positive pairs are on the diagonal """
#         caption_rank = self.sorter(scores)
#         image_rank = self.sorter(scores.t())

#         image_caption_loss = torch.sum(caption_rank.diag()) + torch.sum(image_rank.diag())

#         return image_caption_loss


# class MapRankingLoss(torch.nn.Module):
#     """ Loss function  inspired by mean Average Precision """
#     def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None):
#         super(MapRankingLoss, self).__init__()

#         self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

#     def forward(self, output, target):
#         # Compute map for each classes
#         map_tot = 0
#         for c in range(target.size(1)):
#             gt_c = target[:, c]

#             if torch.sum(gt_c) == 0:
#                 continue
#             rank_pred = self.sorter(output[:, c].unsqueeze(0)).view(-1)
#             rank_pos = rank_pred * gt_c

#             map_tot += torch.sum(rank_pos)

#         return map_tot


# class SpearmanLoss(torch.nn.Module):
#     """ Loss function  inspired by spearmann correlation.self
#     Required the trained model to have a good initlization.
#     Set lbd to 1 for a few epoch to help with the initialization.
#     """
#     def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, lbd=0):
#         super(SpearmanLoss, self).__init__()
#         self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

#         self.criterion_mse = torch.nn.MSELoss()
#         self.criterionl1 = torch.nn.L1Loss()

#         self.lbd = lbd

#     def forward(self, mem_pred, mem_gt, pr=False):
#         rank_gt = get_rank(mem_gt)

#         rank_pred = self.sorter(mem_pred.unsqueeze(
#             0)).view(-1)

#         return self.criterion_mse(rank_pred, rank_gt) + self.lbd * self.criterionl1(mem_pred, mem_gt)

class approxNDCGLoss(nn.Module):
    def __init__(self):
        super(approxNDCGLoss, self).__init__()
        return

    def forward(self, y_pred, y_true, eps=0.0005, padded_value_indicator=-1, alpha=1.):
        """
        Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
        Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param alpha: score difference weight used in the sigmoid function
        :return: loss value, a torch.Tensor
        """
        
            # classifier_lpm = classifier_lpm.cuda()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("ypred")
        # print(y_pred)
        # print("ytrue")
        # print(y_true)
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        # padded_mask = y_true == padded_value_indicator
        # y_pred[padded_mask] = float("-inf")
        # y_true[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
        scores_diffs[~padded_pairs_mask] = 0.
        approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
        approx_D = torch.log2(1. + approx_pos)
        approx_NDCG = torch.sum((G / approx_D), dim=-1)

        return torch.mean(approx_NDCG)

class LambdaLoss(nn.Module):
    def __init__(self):
        super(LambdaLoss, self).__init__()
        return

    def forward(self, y_pred, y_true, eps=0.00001, padded_value_indicator=-1, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="sum", reduction_log="binary"):
        """
        LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
        Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
        :param k: rank at which the loss is truncated
        :param sigma: score difference weight used in the sigmoid function
        :param mu: optional weight used in NDCGLoss2++ weighing scheme
        :param reduction: losses reduction method, could be either a sum or a mean
        :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
        :return: loss value, a torch.Tensor
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        # padded_mask = y_true == padded_value_indicator
        # y_pred[padded_mask] = float("-inf")
        # y_true[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if weighing_scheme is None:
            weights = 1.
        else:
            weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        if reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        if reduction == "sum":
            loss = -torch.sum(masked_losses)
        elif reduction == "mean":
            loss = -torch.mean(masked_losses)
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss

import torch.nn.functional as F
class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()
        return

    def forward(self, y_pred, y_true, eps=0.0005, padded_value_indicator=-1):
        """
        ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        # mask = y_true == padded_value_indicator
        # y_pred[mask] = float('-inf')
        # y_true[mask] = float('-inf')

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

class ListMLELoss(nn.Module):
    def __init__(self):
        super(ListMLELoss, self).__init__()
        return
        
    def forward(self, y_pred, y_true, eps=0.00001, padded_value_indicator=-1):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max


        return torch.mean(torch.sum(observation_loss, dim=1))



class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))







# def approxNDCGLoss(y_pred, y_true, eps=0.0005, padded_value_indicator=-1, alpha=1.):
#     """
#     Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
#     Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
#     :param y_pred: predictions from the model, shape [batch_size, slate_length]
#     :param y_true: ground truth labels, shape [batch_size, slate_length]
#     :param eps: epsilon value, used for numerical stability
#     :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
#     :param alpha: score difference weight used in the sigmoid function
#     :return: loss value, a torch.Tensor
#     """
#     device = y_pred.device
#     y_pred = y_pred.clone()
#     y_true = y_true.clone()

#     # padded_mask = y_true == padded_value_indicator
#     # y_pred[padded_mask] = float("-inf")
#     # y_true[padded_mask] = float("-inf")

#     # Here we sort the true and predicted relevancy scores.
#     y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
#     y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

#     # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
#     true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
#     true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
#     padded_pairs_mask = torch.isfinite(true_diffs)
#     padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

#     # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
#     true_sorted_by_preds.clamp_(min=0.)
#     y_true_sorted.clamp_(min=0.)

#     # Here we find the gains, discounts and ideal DCGs per slate.
#     pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
#     D = torch.log2(1. + pos_idxs.float())[None, :]
#     maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
#     G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

#     # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
#     scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
#     scores_diffs[~padded_pairs_mask] = 0.
#     approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
#     approx_D = torch.log2(1. + approx_pos)
#     approx_NDCG = torch.sum((G / approx_D), dim=-1)

#     return -torch.mean(approx_NDCG)


class Label_Correlation_Loss(nn.Module):
    def __init__(self):
        super(Label_Correlation_Loss, self).__init__()

    def forward(self,pred_Y, true_Y):
        return label_correlation_loss(pred_Y, true_Y)

# Label correlation aware loss function
class DIYloss(nn.Module):
    def __init__(self):
        super(DIYloss, self).__init__()
        return
    def forward(self, pred_Y, true_Y):
        mseLoss = nn.MSELoss(reduction='none')
        pred_Y = torch.sigmoid(pred_Y)
        n_one_true = int(torch.sum(true_Y))
        n_zero_true = true_Y.shape[1] - n_one_true
        nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
        zero_index = torch.nonzero(true_Y[0] == 0).reshape(
            -1)
        Ei = 0
        if n_one_true == 0:
            Ei = (pred_Y[0] ** 2).mean()
        else:
            for k in range(n_one_true):
                for l in range(n_zero_true):
                    Ei = mseLoss((1 + pred_Y[0][zero_index[l]]),
                                 pred_Y[0][nonzero_index[k]]) + Ei
            Ei = 1 / (n_one_true * n_zero_true) * Ei
        return Ei

        return loss


def label_correlation_loss(pred_Y, true_Y):
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
    zero_index = torch.nonzero(true_Y[0] == 0).reshape(-1)
    Ei = 0
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.exp( pred_Y[0][zero_index[0][l]]) + Ei
        Ei = 1 / (n_zero_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = torch.exp(-(pred_Y[0][nonzero_index[0][k]] - pred_Y[0][zero_index[0][l]])) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei

def label_correlation_loss2(pred_Y, true_Y):
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
    zero_index = torch.nonzero(true_Y[0] == 0).reshape(-1)
    Ei = 0
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.exp(pred_Y[0][zero_index[l]] - 1) + Ei
        Ei = 1 / (n_zero_true) * Ei
    elif n_zero_true == 0:
        for l in range(n_one_true):
            Ei = torch.exp(-pred_Y[0][nonzero_index[l]]) + Ei
        Ei = 1 / (n_one_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = torch.exp(-(pred_Y[0][nonzero_index[k]] - pred_Y[0][zero_index[l]])) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei

def label_correlation_loss2_old(pred_Y, true_Y):
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.where(true_Y > 0)# nonzero_index = torch.nonzero(true_Y)
    zero_index = torch.where(true_Y == 0)
    Ei = 0
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.exp(pred_Y[0][zero_index[1][l]] - 1) + Ei
        Ei = 1 / (n_zero_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = torch.exp(-(pred_Y[0][nonzero_index[1][k]] - pred_Y[0][zero_index[1][l]])) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei


# Label correlation aware loss function
def label_correlation_DIYloss(pred_Y, true_Y):
    mseLoss = nn.MSELoss()
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.nonzero(true_Y[0] > 0).reshape(-1)
    zero_index = torch.nonzero(true_Y[0] == 0).reshape(-1)
    Ei = 0
    if n_one_true == 0:
        Ei = (pred_Y[0] ** 2).mean()
    elif n_zero_true == 0:
        Ei = ((pred_Y[0]-1) ** 2).mean()
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = mseLoss((1 + pred_Y[0][zero_index[l]]), pred_Y[0][nonzero_index[k]]) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    return Ei


def label_correlation_DIYloss_old(pred_Y, true_Y):
    mseLoss = nn.MSELoss(reduction='sum')
    pred_Y = torch.sigmoid(pred_Y)
    n_one_true = int(torch.sum(true_Y))
    n_zero_true = true_Y.shape[1] - n_one_true
    nonzero_index = torch.where(true_Y > 0) # nonzero_index = torch.nonzero(true_Y)
    zero_index = torch.where(true_Y == 0)
    Ei = 0
    print(nonzero_index)
    if n_one_true == 0:
        for l in range(n_zero_true):
            Ei = torch.pow(pred_Y[0][zero_index[1][l]], 2) + Ei
        Ei = 1 / (n_zero_true) * Ei
    else:
        for k in range(n_one_true):
            for l in range(n_zero_true):
                Ei = mseLoss((1 + pred_Y[0][zero_index[1][l]]), pred_Y[0][nonzero_index[1][k]]) + Ei
        Ei = 1 / (n_one_true * n_zero_true) * Ei
    # print(n_one_true * n_zero_true)
    return Ei


def label_correlation_loss_batch(pred_Y, true_Y):
    Ei = 0
    for i in range(true_Y.shape[0]):
        n_one_true = int(torch.sum(true_Y[i]))
        n_zero_true = true_Y.shape[1] - n_one_true
        nonzero_index = torch.nonzero(true_Y[i] > 0)
        zero_index = torch.nonzero(true_Y[i] == 0)
        temp = 0
        for k in range(n_one_true):
            for l in range(n_zero_true):
                temp = torch.exp(-(pred_Y[i][nonzero_index[0][k]] - pred_Y[i][zero_index[0][l]])) +temp
        Ei = 1/(n_one_true * n_zero_true) * temp + Ei
    return Ei
