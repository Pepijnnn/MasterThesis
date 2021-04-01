# -*- coding:utf-8 -*- 

# Deep Streaming Label Learning
# Pepijn Sibbes adapted

import torch.nn as nn
import torch
import torch.nn.functional as F

class KnowledgeDistillation(nn.Module):
    def __init__(self, hyper_params):
        super(KnowledgeDistillation, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.KD_input_dim, hyper_params.KD_output_dim),
            torch.nn.Dropout(hyper_params.KD_dropout),
            nn.ReLU(),
        )
    def forward(self, input):
        return self.W_m(input)

class IntegratedDSLL(nn.Module):
    def __init__(self, hyper_params):
        super(IntegratedDSLL, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_hidden2),
        )
        
        self.transformation = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_output_dim, hyper_params.label_mapping_output_dim * 4),
        )
        self.seniorStudent = nn.Sequential(
            nn.Linear((hyper_params.classifier_hidden2 + hyper_params.label_mapping_output_dim * 4),
                      hyper_params.label_representation_hidden1),
            nn.ReLU(),
            nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_output_dim),
        )
    def forward(self, x, y_mapping):
        x_feature_kd = self.W_m(x)
        y_transformation = self.transformation(y_mapping)
        y_new_prediction = self.seniorStudent(torch.cat((x_feature_kd, y_transformation), 1))
        return y_new_prediction, x_feature_kd, y_transformation, y_new_prediction

# copy weights from loss prediction model when trained on old inputs
class LossPredictionMod(nn.Module):
    def __init__(self, hyper_params):
        super(LossPredictionMod, self).__init__()
        self.Fc1 = nn.Sequential(
            nn.Linear(hyper_params.classifier_hidden2, hyper_params.loss_prediction_hidden),
            nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
        )
        
        self.Fc2 = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_output_dim * 4, hyper_params.loss_prediction_hidden),
            nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
        )
        self.Fc3 = nn.Sequential(
            nn.Linear(hyper_params.label_representation_output_dim, hyper_params.loss_prediction_hidden),
            nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
        )
        
        # * 3 depends of how many layers used
        self.fc_concat = nn.Sequential(
            nn.Linear(hyper_params.loss_prediction_hidden * 3, 1),
            # nn.ReLU(),
        )

    def forward(self, wm_input, tf_input, ss_input):
        W_m_output = self.Fc1(wm_input)
        tf_output = self.Fc2(tf_input)
        ss_output = self.Fc3(ss_input)
        predicted_loss = self.fc_concat(torch.cat((tf_output,W_m_output,ss_output),1)) #,
        # predicted_loss = self.fc_concat(torch.cat((W_m_output,ss_output),1))
        # predicted_loss = self.fc_concat(ss_output)
        return predicted_loss

class BinaryRelevance():
    def __init__(self):
        pass
    def forward(self):
        pass

class ClassifierChain():
    def __init__(self):
        pass
    def forward(self):
        pass


import math
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class _classifier(nn.Module):
    def __init__(self, hyper_params):
        super(_classifier, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            # GELU(),
            # nn.LeakyReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)


class _classifierBatchNorm(nn.Module):
    def __init__(self, hyper_params):
        super(_classifierBatchNorm, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            nn.BatchNorm1d(hyper_params.classifier_hidden1),
            nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)


class _classifier2(nn.Module):
    def __init__(self, hyper_params):
        super(_classifier2, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_hidden1),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_hidden2),
            torch.nn.Dropout(hyper_params.classifier_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.classifier_hidden2, hyper_params.classifier_output_dim),
        )

    def forward(self, input):
        return self.W_m(input)

class _S_label_mapping(nn.Module):
    def __init__(self, hyper_params):
        super(_S_label_mapping,self).__init__()
        self.label_mapping = nn.Sequential(
            nn.Linear(hyper_params.label_mapping_input_dim, hyper_params.label_mapping_hidden1),
            torch.nn.Dropout(hyper_params.label_mapping_dropout),
            nn.ReLU(),
            nn.Linear(hyper_params.label_mapping_hidden1, hyper_params.label_mapping_output_dim),
        )

    def forward(self, input):
        return self.label_mapping(input)

class MarginRankingLoss_learning_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginRankingLoss_learning_loss, self).__init__()
        self.margin = margin
    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        pred_loss = inputs[random]
        fr = int(inputs.size(0)//2)
        to = int(inputs.size(0)//2)
        pred_lossi = inputs[:fr]
        pred_lossj = inputs[to:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:fr]
        target_lossj = target_loss[to:]
        final_target = torch.sign(target_lossi - target_lossj)
        
        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin, reduction='mean')
        


# class _S_label_mapping2(nn.Module):
#     def __init__(self, hyper_params):
#         super(_S_label_mapping2, self).__init__()
#         self.label_mapping = nn.Sequential(
#             nn.Linear(hyper_params.label_mapping_input_dim, hyper_params.label_mapping_hidden1),
#             torch.nn.Dropout(hyper_params.label_mapping_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_mapping_hidden1, hyper_params.label_mapping_hidden2),
#             torch.nn.Dropout(hyper_params.label_mapping_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_mapping_hidden2, hyper_params.label_mapping_output_dim),
#         )

#     def forward(self, input):
        # return self.label_mapping(input)




# class _label_representation(nn.Module):
#     def __init__(self, hyper_params):
#         super(_label_representation, self).__init__()
#         self.label_representation = nn.Sequential(
#             nn.Linear(hyper_params.label_representation_input_dim, hyper_params.label_representation_hidden1),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_output_dim),
#         )

#     def forward(self, input):
#         return self.label_representation(input)


# class _label_representation2(nn.Module):
#     def __init__(self, hyper_params):
#         super(_label_representation2, self).__init__()
#         self.label_representation = nn.Sequential(
#             nn.Linear(hyper_params.label_representation_input_dim, hyper_params.label_representation_hidden1),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_hidden2),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden2, hyper_params.label_representation_output_dim),
#         )
#     def forward(self, input):
#         return self.label_representation(input)

# class _S_label_linear_mapping(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(_S_label_mapping,self).__init__()
#         self.linear_mapping = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#         )

#     def forward(self, input):
#         return self.linear_mapping(input)


# class _BP_ML(nn.Module):
#     def __init__(self, hyper_params):
#         super(_BP_ML, self).__init__()
#         self.W_m = nn.Sequential(
#             nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
#             nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
#         )

#     def forward(self, input):
#         return self.W_m(input)


# class _DNN(nn.Module):
#     def __init__(self, hyper_params):
#         super(_DNN, self).__init__()
#         self.W_m = nn.Sequential(
#             nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
#             torch.nn.Dropout(hyper_params.classifier_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.classifier_hidden1, hyper_params.classifier_output_dim),
#         )

#     def forward(self, input):
#         return self.W_m(input)


# class KnowledgeDistillation_start(nn.Module):
#     def __init__(self, hyper_params):
#         super(KnowledgeDistillation_start, self).__init__()
#         self.W_m = nn.Sequential(
#             nn.Linear(hyper_params.KD_input_dim, hyper_params.KD_output_dim),
#             torch.nn.Dropout(hyper_params.KD_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.KD_output_dim, hyper_params.label_mapping_output_dim),
#             nn.ReLU(),
#         )
#     def forward(self, input):
#         return self.W_m(input)


# class IntegratedModel_3net(nn.Module):
#     def __init__(self, hyper_params):
#         super(IntegratedModel_3net, self).__init__()
#         self.W_m = nn.Sequential(
#             nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
#             torch.nn.Dropout(hyper_params.classifier_dropout),
#             nn.ReLU(),
#         )
#         self.label_mapping = nn.Sequential(
#             nn.Linear(hyper_params.label_mapping_input_dim, hyper_params.label_mapping_hidden1),
#             torch.nn.Dropout(hyper_params.label_mapping_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_mapping_hidden1, hyper_params.label_mapping_output_dim),
#         )
#         self.representation = nn.Sequential(
#             nn.Linear((hyper_params.classifier_hidden1 + hyper_params.label_mapping_output_dim),
#                       hyper_params.label_representation_hidden1),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_hidden2),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden2, hyper_params.label_representation_output_dim),
#         )
#     def forward(self, x, y_m):
#         x_feature_kd = self.W_m(x)
#         y_new_mapping = self.label_mapping(y_m.sigmoid())
#         y_new_prediction = self.representation(torch.cat((x_feature_kd, y_new_mapping), 0))
#         return y_new_prediction

# class IntegratedModel_mapping(nn.Module):
#     def __init__(self, hyper_params):
#         super(IntegratedModel_mapping, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
#             torch.nn.Dropout(hyper_params.classifier_dropout),
#             nn.ReLU(),
#         )
#         self.representation = nn.Sequential(
#             nn.Linear((hyper_params.classifier_hidden1 + hyper_params.label_mapping_output_dim),
#                       hyper_params.label_representation_hidden1),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_hidden2),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden2, hyper_params.label_representation_output_dim),
#         )
#     def forward(self, x, y_m, mapping_model):
#         x_feature_kd = self.main(x)
#         y_new_mapping = mapping_model(y_m)
#         y_new_mapping = y_new_mapping.sigmoid()
#         y_new_prediction = self.representation(torch.cat((x_feature_kd, y_new_mapping), 1))
#         return y_new_prediction



# class IntegratedModel(nn.Module):
#     def __init__(self, hyper_params):
#         super(IntegratedModel, self).__init__()
#         self.W_m = nn.Sequential(
#             nn.Linear(hyper_params.classifier_input_dim, hyper_params.classifier_hidden1),
#             nn.Dropout(hyper_params.classifier_dropout),
#             nn.ReLU(),
#         )
#         self.representation = nn.Sequential(
#             nn.Linear((hyper_params.classifier_hidden1 + hyper_params.label_mapping_output_dim * 4),
#                       hyper_params.label_representation_hidden1),
#             torch.nn.Dropout(hyper_params.label_representation_dropout),
#             nn.ReLU(),
#             nn.Linear(hyper_params.label_representation_hidden1, hyper_params.label_representation_output_dim),
#         )
#         self.mapping_W = nn.Sequential(
#             nn.Linear(hyper_params.label_mapping_output_dim, hyper_params.label_mapping_output_dim * 4),
#             nn.ReLU(),
#         )
#     def forward(self, x, soft_y_new):
#         x_feature_kd = self.W_m(x)
#         soft_y_new = self.mapping_W(soft_y_new)
#         y_new_prediction = self.representation(torch.cat((x_feature_kd, soft_y_new), 1))
#         return y_new_prediction