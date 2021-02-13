'''
Analysis tools.

Summary
-------
The function 'anomaly_score' computes a score for each data item
which can be utilized to identify anomalous/out-of-distribution inputs.
For instance, the entropy of the predicted Bernoulli (binary classification)
or categorical (multi-class problem) distribution might act as such a score.
One can also take one minus the top class probability as a simple score.

The function 'calibration_metrics' computes the classification
accuracies over a whole data set for binned values of the confidence.
This establishes the basis for the reliability diagram and its associated
calibration metrics such as the expected or maximum calibration error.
A frequentist notion of well-calibration can be investigated thereby.

'''

import numpy as np
import torch
import torch.distributions as dist

def anomaly_score(model,
                  data_loader,
                  mode='entropy',
                  no_epochs=1,
                  threshold=0.5,
                  **kwargs):
    '''
    Compute anomaly score.

    Summary
    -------
    An anomaly score is computed for each data point.
    Ideally, it should assume low values for inputs from the training
    distribution and high values for out-of-distribution examples.
    The score is based on the predicted distribution over the classes.
    Here, the entropy or one minus the maximum probability can be chosen.

    Parameters
    ----------
    model : VariationalClassification or PyTorch module
        Logits-predicting model.
    data_loader : PyTorch data loader
        Data-generating loader.
    mode : {'entropy', 'maxprob'}
        Determines the score type.
    no_epochs : int
        Number of epochs.
    threshold : float
        Binary probability threshold.

    '''

    # predictions
    _, probs, _, top_prob = _extract_predict(model,
                                             data_loader,
                                             no_epochs,
                                             threshold,
                                             **kwargs)

    # scores
    if mode == 'entropy':
        score = _entropy(probs)
    elif mode == 'maxprob':
        score = _maxprob_score(top_prob)

    # array
    score = _make_array(score).squeeze()

    return score

def _entropy(probs):
    '''Compute entropy score.'''
    if probs.shape[-1] == 1: # binary classifier
        entropy = dist.Bernoulli(probs=probs).entropy()
    else: # multi-class classifier
        entropy = dist.Categorical(probs=probs).entropy()
    return entropy

def _maxprob_score(top_prob):
    '''Compute maxprob score.'''
    score = 1 - top_prob
    return score

def calibration_metrics(model,
                        data_loader,
                        no_bins=10,
                        no_epochs=1,
                        threshold=0.5,
                        **kwargs):
    '''
    Evaluate calibration metrics.

    Summary
    -------
    Calibration metrics are evaluated on the basis of the reliability diagram.
    The accuracy is determined for groups of samples in confidence bins.
    For a perfectly calibrated model the accuracy would equal the confidence.
    The mismatch can be quantified through various calibration errors.
    Here, the expected and maximum calibration error are computed.

    Parameters
    ----------
    model : VariationalClassification or PyTorch module
        Logits-predicting model.
    data_loader : PyTorch data loader
        Data-generating loader.
    no_bins : int
        Number of confidence bins.
    no_epochs : int
        Number of epochs.
    threshold : float
        Binary probability threshold.

    '''

    # predictions
    labels, _, top_class, top_prob = _extract_predict(model,
                                                      data_loader,
                                                      no_epochs,
                                                      threshold,
                                                      **kwargs)

    # arrays
    labels = _make_array(labels).squeeze()
    top_class = _make_array(top_class).squeeze()
    top_prob = _make_array(top_prob).squeeze()

    # confidence bins and accuracies
    conf_edges = np.linspace(0, 1, no_bins+1)
    binned_conf = (conf_edges[1:] + conf_edges[:-1]) / 2
    binned_acc = np.zeros(no_bins)
    binned_no_samples = np.zeros(no_bins, dtype='int')
    for idx in range(no_bins):
        lower = conf_edges[idx]
        upper = conf_edges[idx+1]
        ids = np.where(np.logical_and(top_prob>=lower, top_prob<upper))[0]
        binned_no_samples[idx] = len(ids)
        binned_acc[idx] = (np.sum(labels[ids] == top_class[ids])) \
                          / binned_no_samples[idx] \
                          if binned_no_samples[idx] != 0 else np.nan

    # calibration errors
    binned_ce, ece, mce = _calibration_errors(binned_conf, binned_acc, binned_no_samples)
    ce_dict = {'CEs': binned_ce, 'ECE': ece, 'MCE': mce}

    return conf_edges, binned_acc, ce_dict

def _calibration_errors(binned_conf, binned_acc, binned_no_samples):
    '''Compute calibration errors (bin-wise, expected and maximum).'''
    binned_ce = np.abs(binned_acc - binned_conf)
    mce = np.max([e for e in binned_ce if not np.isnan(e)])
    ece = np.nansum(binned_no_samples * binned_ce) / np.sum(binned_no_samples)
    return binned_ce, ece, mce

def _extract_predict(model,
                     data_loader,
                     no_epochs=1,
                     threshold=0.5,
                     **kwargs):
    '''
    Extract labels from a data loader and model predictions.

    Summary
    -------
    This function extracts the model predictions
    and ground truth labels from a data loader.

    Parameters
    ----------
    model : VariationalClassification or PyTorch module
        Logits-predicting model.
    data_loader : PyTorch data loader
        Data-generating loader.
    no_epochs : int
        Number of epochs.
    threshold : float
        Binary probability threshold.

    '''

    # device
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # eval mode
    if hasattr(model, 'train'):
        model.train(False)

    # labels and probabilities
    labels_list = []
    probs_list = []
    with torch.no_grad():
        for epoch_idx in range(no_epochs):
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if hasattr(model, 'predict_proba'): # VariationalClassification
                    batch_probs = model.predict_proba(X_batch, **kwargs)
                else: # PyTorch module
                    batch_logits = model(X_batch)
                    if batch_logits.shape[-1] == 1: # binary classifier
                        batch_probs = torch.sigmoid(batch_logits)
                    else: # multi-class classifier
                        batch_probs = torch.softmax(batch_logits, dim=-1)
                labels_list.append(y_batch)
                probs_list.append(batch_probs)
    labels = torch.cat(labels_list, dim=0)
    probs = torch.cat(probs_list, dim=0)

    # top class and probability
    if probs.shape[-1] == 1: # binary classifier
        top_class = (probs >= threshold).int()
        top_prob = torch.where(top_class==1, probs, 1-probs)
    else: # multi-class classifier
        top_prob, top_class = torch.topk(probs, k=1, dim=1)

    return labels, probs, top_class, top_prob

def _make_array(tensor):
    '''Transform tensor into array.'''
    array = tensor.data.cpu().numpy()
    return array

