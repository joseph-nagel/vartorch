'''
Confidence well-calibration.

Summary
-------
The function 'accuracy_vs_confidence' computes the classification
accuracies for binned values of the confidence score.
This establishes a notion of well-calibration from a frequentist perspective.

'''

import numpy as np
import torch

def accuracy_vs_confidence(model,
                           data_loader,
                           no_bins=10,
                           likelihood_type='Categorical',
                           no_samples=None,
                           threshold=0.5):
    '''Compute accuracies in confidence bins.'''

    # likelihood type
    if hasattr(model, 'likelihood_type'):
        likelihood_type = model.likelihood_type

    # sampling and training mode
    if hasattr(model, 'sample'):
        if no_samples is None:
            model.sample(False)
        else:
            model.sample(True)
    if hasattr(model, 'train'):
        model.train(False)

    # predictions
    labels = []
    top_prob = []
    top_class = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            # probabilities (without sampling)
            if no_samples is None:
                batch_logits = model.predict(X_batch)
                if likelihood_type == 'Categorical':
                    batch_probs = torch.softmax(batch_logits, dim=1)
                elif likelihood_type == 'Bernoulli':
                    batch_probs = torch.sigmoid(batch_logits)
            # probabilities (with sampling)
            else:
                sampled_logits = model.predict(X_batch, no_samples)
                if likelihood_type == 'Categorical':
                    sampled_probs = torch.softmax(sampled_logits, dim=1)
                elif likelihood_type == 'Bernoulli':
                    sampled_probs = torch.sigmoid(sampled_logits)
                batch_probs = torch.mean(sampled_probs, axis=-1)
                # batch_probs = model.predict_proba(X_batch, no_samples)
            # top class and probability
            if likelihood_type == 'Categorical':
                batch_top_prob, batch_top_class = torch.topk(batch_probs, k=1, dim=1)
            elif likelihood_type == 'Bernoulli':
                batch_top_class = (batch_probs >= threshold).int()
                batch_top_prob = torch.where(batch_top_class==1, batch_probs, 1-batch_probs)
            labels.append(y_batch)
            top_prob.append(batch_top_prob)
            top_class.append(batch_top_class)
    labels = torch.cat([_ for _ in labels], dim=0).squeeze().data.numpy()
    top_prob = torch.cat([_ for _ in top_prob], dim=0).squeeze().data.numpy()
    top_class = torch.cat([_ for _ in top_class], dim=0).squeeze().data.numpy()

    # confidence binning & accuracy evaluation
    confidence_edges = np.linspace(0, 1, no_bins+1)
    bin_accuracies = np.zeros(no_bins)
    bin_no_samples = np.zeros(no_bins, dtype='int')
    for idx in range(no_bins):
        lower = confidence_edges[idx]
        upper = confidence_edges[idx+1]
        ids = np.where(np.logical_and(top_prob>=lower, top_prob<upper))[0]
        bin_no_samples[idx] = len(ids)
        bin_accuracies[idx] = np.sum(labels[ids] == top_class[ids]) / len(ids) if len(ids) != 0 else np.nan

    return confidence_edges, bin_accuracies

