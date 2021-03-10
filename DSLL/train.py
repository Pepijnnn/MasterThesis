# -*- coding:utf-8 -*-

# Deep Streaming Label Learning
import sklearn.metrics as metrics
from model import _label_representation, _S_label_mapping, _classifier, _classifier2, IntegratedModel,\
    KnowledgeDistillation,  _classifierBatchNorm, _S_label_mapping2, _DNN, _BP_ML, IntegratedDSLL, LossPredictionMod

from helpers import predictor_accuracy, precision_at_ks, predict, predict_integrated, \
    print_predict, LayerActivations, modify_state_dict

import numpy as np

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
        print_predict(train_Y_new, pred_Y_train, hyper_params)

    if (((hyper_params.currentEpoch + 1) % 5 == 0) | (hyper_params.currentEpoch < 10)):
        print('test performance')
        pred_Y = predict_integrated(classifier, test_X, mapping_test_Y_new)
        # pred_Y = pred_Y.round()
        print_predict(test_Y_new, pred_Y, hyper_params)


def observe_train(hyper_params, classifier, training_losses, train_X, train_Y, test_X, test_Y):
    print('[%d/%d]Loss: %.3f' % (
        hyper_params.currentEpoch + 1, hyper_params.classifier_epoch, np.mean(training_losses)))
    if ((((hyper_params.currentEpoch + 1) % 10) == 0) | ((hyper_params.currentEpoch + 1)
                                                         == hyper_params.classifier_epoch)):
        print('train performance')
        pred_Y_train = predict(classifier, train_X)
        print_predict(train_Y, pred_Y_train, hyper_params)

    if (((hyper_params.currentEpoch + 1) % 5 == 0) | (hyper_params.currentEpoch < 10)):
        print('test performance')
        pred_Y = predict(classifier, test_X)
        print_predict(test_Y, pred_Y, hyper_params)

def train_KD(hyper_params, train_X, train_Y, test_X, test_Y):
    print("train_KD\ninput and output dims")
    hyper_params.KD_input_dim = train_X.shape[1]
    hyper_params.KD_output_dim = train_Y.shape[1]
    print(hyper_params.KD_input_dim)
    print(hyper_params.KD_output_dim)
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
        # print(yhat.shape,y.shape)
        loss = loss_fn(yhat, y)
        # print("dit is los item")
        # print(loss)
        # print(loss.mean())
        # print("this is loss item")
        # Computes gradients
        loss.mean().backward()
        # loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        
        return loss, kd_mid, trans_mid, ss_mid

    # Returns the function that will be called inside the train loop
    return train_step_DSLL

################################################################## DSLL ###################################################################
def train_DSLL_model(hyper_params, featureKD_model, train_X, train_Y, mapping_train_Y_new, train_Y_new, test_X, mapping_test_Y_new, test_Y_new, train_loader):
    hyper_params.classifier_input_dim = train_X.shape[1]
    hyper_params.classifier_output_dim = train_Y.shape[1]
    device = hyper_params.device
    hyper_params.model_name = 'DSLL'
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
    # optimizer = optim.Adam(classifier.parameters(), weight_decay=hyper_params.classifier_L2)
    optimizer = torch.optim.Adam([
        {'params':classifier.W_m.parameters()},   # , 'lr': 0.0001},
        {'params':classifier.seniorStudent.parameters()},
        {'params':classifier.transformation.parameters()},
    ], weight_decay=hyper_params.classifier_L2)

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
        criterion = nn.MultiLabelSoftMarginLoss(reduction ='none') 
    train_step = make_train_DSLL(classifier, criterion, optimizer)

    # lp_criterion = approxNDCGLoss()
    # lp_criterion = LambdaLoss()
    # lp_criterion = ListNetLoss()
    lp_criterion = RMSELoss()
    # lp_criterion = ListMLELoss()

    
    classifier_lpm = LossPredictionMod(hyper_params)
    # optimizer2 = optim.Adam(classifier_lpm.parameters(), weight_decay=hyper_params.classifier_L2)
    optimizer2 = torch.optim.Adam([
            {'params':classifier_lpm.Fc1.parameters()},   # , 'lr': 0.0001},
            {'params':classifier_lpm.Fc2.parameters()},
            {'params':classifier_lpm.Fc3.parameters()},
            {'params':classifier_lpm.fc_concat.parameters()},
        ], weight_decay=hyper_params.classifier_L2)

     
    if torch.cuda.is_available():
        classifier_lpm = classifier_lpm.cuda()
    activation = {}
    # def get_activation(name):
    #     # print("INN")
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #         print("INN")
    #         print(output.detach())
    #     return hook


    training_losses = []

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
            loss ,kd_mid, trans_mid, ss_mid = train_step(x_batch, y_mapping, y_batch)
            batch_losses.append(loss.mean().item()) #.mean()

            # print(kd_mid, trans_mid, ss_mid)
            # copy weights from IntegratedDSLL model to loss prediction model
            # classifier.eval()


            optimizer.zero_grad()
            optimizer2.zero_grad()
            # Makes predictions detach old loss function (only update over loss prediction module)
            kd_mid, trans_mid, ss_mid = kd_mid.detach(), trans_mid.detach(), ss_mid.detach()
            predicted_loss = classifier_lpm(kd_mid, trans_mid, ss_mid)

            loss2 = lp_criterion(predicted_loss, loss.unsqueeze(1).detach())
            loss3 = loss.mean().detach() + loss2
            # Computes gradients and updates model
            loss3.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            if len(batch_losses) % 10 == 0:
                # print(predicted_loss,loss.unsqueeze(1))
                # print(f"loss prediction loss: {loss3}")
                # true loss number is number in row (first is highest)
                ndcg_true = np.asarray(loss.unsqueeze(1).cpu().detach().numpy())
                ndcg_seq = sorted(ndcg_true)
                ndcg_index = np.asarray([ndcg_seq.index(v) for v in ndcg_true])[..., np.newaxis]
                # compare rank with score higher score is higher confidence so needs to match true loss rank
                ndcg_score =  np.asarray(predicted_loss.cpu().detach().numpy())
                # right size for the ndcg is (1,batch_size)
                ndcg_index.resize(1,batch_size)
                ndcg_score.resize(1,batch_size)
                # ndcg at half of batch size 
                batch_ndcg = metrics.ndcg_score(ndcg_index,ndcg_score, k=int(batch_size/2))
                # TRAIN NDCG SAVED MOVED FOR TEST
                # ndcg_saved.append(batch_ndcg)
                
                # print(f"batch NDCG real score: {batch_ndcg}")

                ##### TEST NDCG #####
                classifier.eval()
                test_loss ,kd_mid_test, trans_mid_test, ss_mid_test = train_step( torch.from_numpy(test_X).float().to(device),\
                                            torch.from_numpy(mapping_test_Y_new).float().to(device),\
                                            torch.from_numpy(test_Y_new).float().to(device))
                kd_mid_test, trans_mid_test, ss_mid_test = kd_mid_test.detach(), trans_mid_test.detach(), ss_mid_test.detach()
                predicted_loss_test = classifier_lpm(kd_mid_test, trans_mid_test, ss_mid_test)
                # print(f"Test loss size: {test_loss.shape[0]}, with mean of {test_loss.mean()}")
                ndcg_true = np.asarray(test_loss.unsqueeze(1).cpu().detach().numpy())
                ndcg_seq = sorted(ndcg_true)
                ndcg_index = np.asarray([ndcg_seq.index(v) for v in ndcg_true])[..., np.newaxis]
                # compare rank with score higher score is higher confidence so needs to match true loss rank
                ndcg_score =  np.asarray(predicted_loss_test.cpu().detach().numpy())
                # right size for the ndcg is (1,batch_size)
                ndcg_index.resize(1,test_loss.shape[0])
                ndcg_score.resize(1,test_loss.shape[0])
                # ndcg at 10 percent of test size
                test_ndcg = metrics.ndcg_score(ndcg_index,ndcg_score, k=int(test_loss.shape[0]*0.1))
                ndcg_saved.append(test_ndcg)
                print(f"Test NDCG: {test_ndcg}")
                classifier.train()
                # exit()
                
        
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        observe_train_DSLL(hyper_params, classifier, training_losses, train_X, mapping_train_Y_new, train_Y_new, test_X,
                           mapping_test_Y_new, test_Y_new)

    x_axis = [i for i in range(len(ndcg_saved))]
    import matplotlib.pyplot as plt
    plt.plot(x_axis, ndcg_saved)
    plt.xlabel('instances', fontsize=18)
    plt.ylabel('ndcg', fontsize=16)
    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(x_axis, ndcg_saved)
    print(slope)
    print(np.mean(ndcg_saved))
    plt.plot(np.unique(x_axis), np.poly1d(np.polyfit(x_axis, ndcg_saved, 1))(np.unique(x_axis)))
    plt.show()
    # exit()

    print('complete the training')
    return classifier



def train_S_label_mapping(hyper_params, train_Y, train_Y_new, test_Y, test_Y_new):
    hyper_params.label_mapping_input_dim = train_Y.shape[1]
    hyper_params.label_mapping_output_dim = train_Y_new.shape[1]
    title1 = ['train_S_label_mapping', 'input_dim={}, '.format(hyper_params.label_mapping_input_dim),
              'output_dim={}, '.format(hyper_params.label_mapping_output_dim),
              'dropout rate={}, '.format(hyper_params.label_mapping_dropout),
              'hidden1={}, '.format(hyper_params.label_mapping_hidden1),
              'hidden2={}, '.format(hyper_params.label_mapping_hidden2),
              'epoch={}'.format(hyper_params.label_mapping_epoch),  'L2={}'.format(hyper_params.label_mapping_L2)
              ]
    print(title1)
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


def train_label_representation(hyper_params, train_X, mapping_soft_train_Y_new, train_Y_new, test_X, mapping_soft_test_Y_new, test_Y_new):
    hyper_params.label_representation_input_dim = train_X.shape[1] + mapping_soft_train_Y_new.shape[1]
    hyper_params.label_representation_output_dim = train_Y_new.shape[1]
    label_representation = _label_representation(hyper_params)
    if torch.cuda.is_available():
        label_representation = label_representation.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        label_representation = nn.DataParallel(label_representation, device_ids=[0, 1])


    optimizer_label_repre = optim.Adam(label_representation.parameters(),
                                       weight_decay=hyper_params.label_representation_L2)
    criterion_label_repre = nn.MultiLabelSoftMarginLoss()  # nn.MultiLabelSoftMarginLoss()
    train_input = np.hstack((train_X, mapping_soft_train_Y_new))
    train_output_true = train_Y_new
    test_input = np.hstack((test_X, mapping_soft_test_Y_new))
    test_output_true = test_Y_new
    for epoch in range(hyper_params.label_representation_epoch):
        losses = []
        for i, sample in enumerate(train_input):
            if torch.cuda.is_available():
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1).cuda()
                labelsv = Variable(torch.FloatTensor(train_output_true[i])).view(1, -1).cuda()
            else:
                inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
                labelsv = Variable(torch.FloatTensor(train_output_true[i])).view(1, -1)

            output = label_representation(inputv)
            loss = criterion_label_repre(output, labelsv)

            optimizer_label_repre.zero_grad()
            loss.backward()
            optimizer_label_repre.step()
            losses.append(loss.data.mean().item())
    print('complete the label representation')
    return label_representation

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
        mseLoss = nn.MSELoss()
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