import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score, f1_score
from sklearn.linear_model import LogisticRegression



def train_BR_CC(train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest, seed_pool, seednr):

    # Fit an independent logistic regression model for each class using the OnevsRest classifier
    train_Y = np.concatenate((train_Y,train_Y_rest),axis=1)
    test_Y = np.concatenate((test_Y,test_Y_rest),axis=1)

    base_lr = LogisticRegression()
    ovr = OneVsRestClassifier(base_lr)
    ovr.fit(train_X, train_Y)
    Y_pred_ovr = ovr.predict(test_X)
    ovr_f1_score = f1_score(test_Y, Y_pred_ovr, average='micro')

    # Fit an ensemble of logistic regression classifier chains and take the
    # take the average prediction of all the chains.
    chains = [ClassifierChain(base_lr, order='random', random_state=i) for i in range(10)]
    for chain in chains:
        chain.fit(train_X, train_Y)

    Y_pred_chains = np.array([chain.predict(test_X) for chain in chains])

    chain_f1_scores = [f1_score(test_Y, Y_pred_chain >= .5, average='micro') for Y_pred_chain in Y_pred_chains]

    Y_pred_ensemble = Y_pred_chains.mean(axis=0)

    ensemble_f1_score = f1_score(test_Y, Y_pred_ensemble >= .5, average='micro')

    model_scores = [ovr_f1_score] + chain_f1_scores
    model_scores.append(ensemble_f1_score)

    model_names = ('Independent', 'Chain 1', 'Chain 2', 'Chain 3', 'Chain 4', 'Chain 5', 'Chain 6', 'Chain 7', 'Chain 8', 'Chain 9', 'Chain 10', 'Ensemble')

    x_pos = np.arange(len(model_names))

    # Plot the F1 similarity scores for the independent model, each of the
    # chains, and the ensemble (note that the vertical axis on this plot does
    # not begin at 0).

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(True)
    ax.set_title('Classifier Chain Ensemble Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation='vertical')
    ax.set_ylabel('Micro-F1')
    ax.set_ylim([min(model_scores) * .9, max(model_scores) * 1.1])
    colors = ['r'] + ['b'] * len(chain_f1_scores) + ['g']
    ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
    plt.tight_layout()
    plt.show()
    

    exit()