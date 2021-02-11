import copy
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from src.utils import args, load_data, logging
from sklearn.metrics import f1_score
from src.models import load_model
from sklearn.metrics import hamming_loss, average_precision_score, precision_score

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

@torch.no_grad()
def valid_step(model, criterion, val_loader):
    model.eval()
    avg_loss, avg_acc, avg_f1, count, avg_prec = 0.0, 0.0, 0.0, 0, 0.0
    for i, (x_imgs, labels) in enumerate(val_loader):
        # forward pass
        x_imgs, labels = x_imgs.to(args.device), labels.type(torch.FloatTensor).to(args.device)
        outputs = model(x_imgs)
        loss = criterion(outputs.squeeze(1), labels)
        # gather statistics
        avg_loss += loss.item()
        #_, preds = torch.max(outputs, 1)
        #avg_acc += torch.sum(preds == labels.data).item()
        a, b = outputs.cpu().detach().numpy().round(), labels.cpu().numpy()
        if args.model == 'no_img':
            a = a.squeeze(1)
        
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        sigmoided = np.vectorize(sigmoid)
        a = sigmoided(a)

        a = a.round()

        if args.dataset == "iris" or args.dataset == "wine":
            avg_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            avg_acc += torch.sum(preds == labels.data).item()
        else:
            accuracy_score = [f1_score(tru,pred,average='micro') for tru,pred in zip(a,b)]
            # accuracy_score = []
            # for tru,pred in zip (a, b):
            #     accuracy_score.append(f1_score(tru,pred,average='micro'))

            avg_f1 += np.mean(accuracy_score)
            # avg_acc += np.mean([hamming_score(true,pred) for true, pred in zip(a,b)])
            hamlosacc = np.mean([hamming_loss(true,pred) for true, pred in zip(a,b)])
            avg_acc += 1-hamlosacc
            # print(hamlosacc)
            
            # print(a,b)
            to_mean = [precision_score(true,pred,average='micro')for true, pred in zip(a,b)]
            # print(to_mean)
            # where_are_NaNs = np.isnan(to_mean)
            # to_mean[where_are_NaNs] = 0
            avg_prec += np.mean(np.nan_to_num(to_mean))
            # print(a,b,accuracy_score,avg_prec/count)
        count+=1
    if args.dataset == "iris" or args.dataset == "wine":
        return {'loss': avg_loss / len(val_loader), 'accuracy': avg_acc / len(val_loader.dataset), 'f1':0}
    else:
        
        return {'loss': avg_loss / len(val_loader), 'avg_prec': avg_acc / count,  'f1': avg_f1/count, 'accuracy':avg_prec/count} #avg_acc / len(val_loader.dataset)


def train_step(model, criterion, optimizer, train_loader, g, epoch):
    if args.model == 'densenet121_n':
        model, class_modules = model
    model.train()
    avg_loss, avg_acc, avg_f1, count, avg_prec = 0.0, 0.0, 0.0, 0, 0.0
    for i, (x_imgs, labels) in enumerate(train_loader):
        # print(i)
        g+=1
        if g == 0:
            print(x_imgs[0].shape)
            plt.grid()
            plt.imshow(x_imgs[0][0, :, :]) # plt.imshow(x_imgs[0][0, :, :])
            plt.show()
            g+=1
            # exit()
            
        optimizer.zero_grad()
        # forward pass
        x_imgs, labels = x_imgs.to(args.device), labels.type(torch.FloatTensor).to(args.device)
        probs = model(x_imgs)
        # print(labels.shape)
        # print(probs.shape)
        # print(probs.squeeze(1).squeeze(1).shape)
        
        loss = criterion(probs.squeeze(1), labels)
        # back-prop
        loss.backward()
        optimizer.step()
        # gather statistics
        avg_loss += loss.item()

        # print(probs.squeeze(1).round(), labels)
        a, b = probs.cpu().detach().numpy(), labels.cpu().numpy()
        if args.model == 'no_img':
            a = a.squeeze(1)

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        sigmoided = np.vectorize(sigmoid)
        a = sigmoided(a)

        a = a.round()
        # if i%15 == 0:
        #     print(loss)
        #     print(a,b)
        
        # print(b)
        # for i in b:
        #     i[0] = 1
        # print(a,b)
        
        if args.dataset == "iris" or args.dataset == "wine":
            avg_loss += loss.item()
            _, preds = torch.max(probs, 1)
            avg_acc += torch.sum(preds == labels.data).item()
        else:
            accuracy_score = [f1_score(tru,pred,average='micro') for tru,pred in zip(a,b)]
            # for tru,pred in zip (a, b):
            #     accuracy_score.append(f1_score(tru,pred,average='micro'))
            # average_precision_score
            avg_f1 += np.mean(accuracy_score)
            # avg_acc += np.mean([hamming_score(true,pred) for true, pred in zip(a,b)])
            hamlosacc = np.mean([hamming_loss(true,pred) for true, pred in zip(a,b)])
            avg_acc += 1-hamlosacc
            # print(hamlosacc)
            
            # print([average_precision_score(true,pred)for true, pred in zip(a,b)])
            # print(a,b)
            to_mean = [precision_score(true,pred,average='micro')for true, pred in zip(a,b)]
            # print(to_mean)
            # where_are_NaNs = np.isnan(to_mean)
            # to_mean[where_are_NaNs] = 0
            avg_prec += np.mean(np.nan_to_num(to_mean))

        count+=1
        if epoch == 12:
            print(labels[0])
            print(probs[0])
            print(a,b)
            # print(np.mean(accuracy_score))
            exit()
        # if count == int(len(train_loader)/100):
        #     print(f"Length is {len(train_loader)}")
        # break

    if args.dataset == "iris" or args.dataset == "wine":
        return {'loss': avg_loss / len(train_loader), 'accuracy': avg_acc / len(train_loader.dataset), 'f1':0}
    else:
        return {'loss': avg_loss / len(train_loader), 'avg_prec': avg_acc/count, 'f1': avg_f1/count, 'accuracy':avg_prec/count} #avg_acc / len(train_loader.dataset)




def opt_selection(model, opt=args.opt):
    if opt=='Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.0001)
    elif opt=='Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif opt=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise NotImplementedError
    return optimizer

@torch.no_grad()
def valid_step_single(model, criterion, val_loader):
    if args.model == 'ownnet':
        model, class_modules = model
    model.eval()
    avg_loss, avg_acc, avg_f1, count, avg_prec = 0.0, 0.0, 0.0, 0, 0.0
    for i, (x_imgs, labelss) in enumerate(val_loader):
        for j in range(10):
            labels = labelss[...,j]
            # forward pass
            x_imgs, labels = x_imgs.to(args.device), labels.type(torch.FloatTensor).to(args.device)
            outputs = class_modules[j](model(x_imgs))
            loss = criterion(outputs.squeeze(), labels)
            # gather statistics
            avg_loss += loss.item()
            #_, preds = torch.max(outputs, 1)
            #avg_acc += torch.sum(preds == labels.data).item()
            a, b = outputs.cpu().detach().numpy().round(), labels.cpu().numpy()
            a[a < 0] = 0 
            a[a > 1] = 1
            a = a.round()
            # for i in b:
            #     i[0] = 1
            # avg_acc += hamming_score(a,b)
            # accuracy_score = []

            # for tru,pred in zip (a, b):
            #     accuracy_score.append(f1_score(tru,pred,average='micro'))

            # avg_f1 += np.mean(accuracy_score)
            avg_f1 += (a == b).sum()
            count+=1

    return {'loss': avg_loss / len(val_loader), 'accuracy': avg_f1 / count,  'f1': avg_f1/count} #avg_acc / len(val_loader.dataset)

def train_step_single(model, criterion, optimizer, train_loader, g, epoch):
    if args.model == 'ownnet':
        model, class_modules = model
    model.train()
    avg_loss, avg_acc, avg_f1, count, av_prec = 0.0, 0.0, 0.0, 0, 0.0
    for i, (x_imgs, labelss) in enumerate(train_loader):
        for j in range(10):
            # print(labelss)
            labels = labelss[...,j]
            # print(labels)
            optimizer.zero_grad()
            # forward pass
            x_imgs, labels = x_imgs.to(args.device), labels.type(torch.FloatTensor).to(args.device)
            probs = class_modules[j](model(x_imgs))
            # print(probs, labels)
            # labels = labels[j]
            loss = criterion(probs.squeeze(), labels) #.unsqueeze_(1)
            # print(probs.squeeze(), labels, loss)
            # back-prop
            loss.backward()
            optimizer.step()
            # gather statistics
            avg_loss += loss.item()

            a, b = probs.cpu().detach().numpy(), labels.cpu().numpy()
            a[a < 0] = 0 
            a[a > 1] = 1
            a = a.round()
            # print(b)
            # for i in b:
            #     i[0] = 1
            #print(a,b)
            # accuracy_score = []
            avg_f1 += (a == b).sum()

            

            # for tru,pred in zip (a, b):
            #     print(tru.squeeze(),pred)
            #     accuracy_score.append(f1_score(tru,pred,average='micro'))

            # avg_f1 += np.mean(accuracy_score)
            count+=1
            if epoch == 12:
                print(labels)
                print(probs)
                print(a,b)
                # print(np.mean(accuracy_score))
                exit()
            
            # avg_acc += hamming_score(a,b)
    #print(avg_f1, len(train_loader.dataset), count)
    return {'loss': avg_loss / len(train_loader), 'accuracy': avg_f1/count, 'f1': avg_f1/count} #avg_acc / len(train_loader.dataset)



def train_model(model_name='densenet121', opt='Adagrad', dataset='iris', writer=None):
    train_loader, val_loader, test_loader = load_data(dataset)

    # Model selection
    model = load_model(model_name)

    # Optimizer
    if model_name == "ownnet":
        optimizer = opt_selection(model[0], opt)
    else:
        optimizer = opt_selection(model, opt)

    # Loss Criterion
    if dataset == 'mltoy' or dataset == "yeast14c" or dataset == "yeast14c_m":
        # criterion = nn.MultiLabelSoftMarginLoss()
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if model_name == "ownnet":
        criterion = torch.nn.MSELoss()

    best_train, best_val = 0.0, 0.0
    g = 0
    for epoch in range(1, args.epochs+1):
        # Train and Validate
        train_stats = train_step(model, criterion, optimizer, train_loader, g, epoch)
        valid_stats = valid_step(model, criterion, val_loader)
        g+=1

        # Logging
        logging(epoch, train_stats, valid_stats, writer)

        # Keep best model
        # print(train_stats['accuracy'], valid_stats['accuracy'], best_train, best_val)
        if valid_stats['accuracy'] > best_val or (valid_stats['accuracy']==best_val and train_stats['accuracy']>=best_train):
            best_train  = train_stats['accuracy']
            best_val    = valid_stats['accuracy']
            if model_name == "ownnet":
                best_model_weights = copy.deepcopy(model[0].state_dict())
            else:
                best_model_weights = copy.deepcopy(model.state_dict())

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_weights)
    test_stats = valid_step(model, criterion, test_loader)
    # print(train_stats['accuracy'], valid_stats['accuracy'], best_train, best_val)
    print('\nBests Model Accuracies: Train: {:4.2f} | Val: {:4.2f} | Test: {:4.2f}'.format(best_train, best_val, test_stats['accuracy']))

    return model


if __name__ == "__main__":
    pass
