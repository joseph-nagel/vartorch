'''
Analysis tools.

Summary
-------
The function `anomaly_score` computes a score for each data item
which can be utilized to identify anomalous/out-of-distribution inputs.
For instance, the entropy of the predicted Bernoulli (binary classification)
or categorical (multi-class problem) distribution might act as such a score.
One can also take one minus the top class probability as a simple score.

The function `calibration_metrics` computes the classification
accuracies over a whole data set for binned values of the confidence.
This establishes the basis for the reliability diagram and its associated
calibration metrics such as the expected or maximum calibration error.
A frequentist notion of well-calibration can be investigated thereby.

'''

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader


# TODO: change output to tensor
def anomaly_score(
    model: nn.Module,
    data_loader: DataLoader,
    mode: str = 'entropy',
    num_epochs: int = 1,
    threshold: float = 0.5,
    **kwargs: Any
) -> np.ndarray:
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
    model : PyTorch/Lightning module
        Logits-predicting model.
    data_loader : PyTorch data loader
        Data-generating loader.
    mode : {'entropy', 'maxprob'}
        Determines the score type.
    num_epochs : int
        Number of epochs.
    threshold : float
        Binary probability threshold.

    '''

    # turn off train mode
    model.train(False)

    # make predictions
    _, probs, _, top_prob = _extract_predict(
        model,
        data_loader,
        num_epochs,
        threshold,
        **kwargs
    )

    # calculate scores
    if mode == 'entropy':
        score_tensor = _entropy(probs)
    elif mode == 'maxprob':
        score_tensor = _maxprob_score(top_prob)

    # transform to array
    score = _make_array(score_tensor).squeeze()

    return score


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    '''Compute entropy score.'''

    # compute Bernoulli entropy (binary classifier)
    if probs.shape[-1] == 1:
        entropy = dist.Bernoulli(probs=probs).entropy()

    # compute categorical entropy (multi-class classifier)
    else:
        entropy = dist.Categorical(probs=probs).entropy()

    return entropy


def _maxprob_score(top_prob: torch.Tensor) -> torch.Tensor:
    '''Compute maxprob score.'''
    return 1 - top_prob


# TODO: change output to tensors
def calibration_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    num_bins: int = 10,
    num_epochs: int = 1,
    threshold: float = 0.5,
    **kwargs: Any
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
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
    model : PyTorch/Lightning module
        Logits-predicting model.
    data_loader : PyTorch data loader
        Data-generating loader.
    num_bins : int
        Number of confidence bins.
    num_epochs : int
        Number of epochs.
    threshold : float
        Binary probability threshold.

    '''

    # turn off train mode
    model.train(False)

    # make predictions
    labels_tensor, _, top_class_tensor, top_prob_tensor = _extract_predict(
        model,
        data_loader,
        num_epochs,
        threshold,
        **kwargs
    )

    # transform to arrays
    labels = _make_array(labels_tensor).squeeze()
    top_class = _make_array(top_class_tensor).squeeze()
    top_prob = _make_array(top_prob_tensor).squeeze()

    # calculate confidence bins and accuracies
    conf_edges = np.linspace(0, 1, num_bins+1)
    binned_conf = (conf_edges[1:] + conf_edges[:-1]) / 2

    binned_acc = np.zeros(num_bins)
    binned_num_samples = np.zeros(num_bins, dtype='int')

    for idx in range(num_bins):

        lower = conf_edges[idx]
        upper = conf_edges[idx+1]

        ids = np.where(np.logical_and(top_prob >= lower, top_prob < upper))[0]

        binned_num_samples[idx] = len(ids)
        binned_acc[idx] = (np.sum(labels[ids] == top_class[ids])) / binned_num_samples[idx] \
                          if binned_num_samples[idx] != 0 else np.nan

    # calculate calibration errors
    binned_ce, ece, mce = _calibration_errors(
        binned_conf, binned_acc, binned_num_samples
    )

    ce_dict = {
        'CEs': binned_ce,
        'ECE': ece,
        'MCE': mce
    }

    return conf_edges, binned_acc, ce_dict


# TODO: change inputs and outputs to tensors
def _calibration_errors(
    binned_conf: np.ndarray,
    binned_acc: np.ndarray,
    binned_num_samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Compute calibration errors (bin-wise, expected and maximum).'''

    binned_ce = np.abs(binned_acc - binned_conf)

    mce = np.max([e for e in binned_ce if not np.isnan(e)])
    ece = np.nansum(binned_num_samples * binned_ce) / np.sum(binned_num_samples)

    return binned_ce, ece, mce


@torch.no_grad()
def _extract_predict(
    model: nn.Module,
    data_loader: DataLoader,
    num_epochs: int = 1,
    threshold: float = 0.5,
    **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Extract labels from a data loader and model predictions.

    Summary
    -------
    This function extracts the model predictions
    and ground truth labels from a data loader.

    Parameters
    ----------
    model : PyTorch/Lightning module
        Logits-predicting model.
    data_loader : PyTorch data loader
        Data-generating loader.
    num_epochs : int
        Number of epochs.
    threshold : float
        Binary probability threshold.

    '''

    # set device
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # turn off train mode
    model.train(False)

    # gather labels and probabilities
    labels_list = []
    probs_list = []

    for _ in range(num_epochs):
        for x_batch, y_batch in data_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # predict probabilities directly
            if hasattr(model, 'predict_proba'):
                batch_probs = model.predict_proba(x_batch, **kwargs)

            # predict logits first
            else:
                batch_logits = model(x_batch)

                if batch_logits.shape[-1] == 1:  # binary classifier
                    batch_probs = torch.sigmoid(batch_logits)

                else:  # multi-class classifier
                    batch_probs = torch.softmax(batch_logits, dim=-1)

            labels_list.append(y_batch)
            probs_list.append(batch_probs)

    labels = torch.cat(labels_list, dim=0)
    probs = torch.cat(probs_list, dim=0)

    # get top class and probability
    if probs.shape[-1] == 1:  # binary classifier
        top_class = (probs >= threshold).int()
        top_prob = torch.where(top_class==1, probs, 1 - probs)

    else:  # multi-class classifier
        top_prob, top_class = torch.topk(probs, k=1, dim=1)

    return labels, probs, top_class, top_prob


def _make_array(tensor: torch.Tensor) -> np.ndarray:
    '''Transform tensor into array.'''
    return tensor.detach().cpu().numpy()
