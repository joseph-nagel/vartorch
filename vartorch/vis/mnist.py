'''For the MNIST example.'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist


def plot_point_predictions(
    images,
    probs,
    labels=None,
    names=None,
    nrows=3,
    figsize=(5, 6),
    title='Point predictions'
):
    '''Plot point predictions.'''

    entropy = dist.Categorical(probs=probs).entropy()  # compute entropy from probabilities

    num_classes = probs.shape[-1]  # get number of classes

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)

    for idx, (ax1, ax2) in enumerate(axes):

        # image
        image = images[idx,0].numpy()
        ax1.imshow(image.clip(0, 1), cmap='gray')
        if labels is not None and names is not None:
            ax1.set_title(f'{names[labels[idx]]}')
        ax1.set(xticks=[], yticks=[], xlabel='', ylabel='')

        # probabilities
        ax2.bar(np.arange(num_classes), probs.detach().cpu().numpy()[idx])
        ax2.set_title('$\pi(c|x,\hat{w})$')
        ax2.set(xticks=np.arange(num_classes), ylim=(0, 1), xlabel='c')
        ax2.text(0, 0.75, 'entropy: {:.2f}'.format(entropy[idx]), alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_post_predictions(
    images,
    sampled_probs,
    labels=None,
    names=None,
    nrows=3,
    figsize=(8, 6),
    title='Posterior predictions'
):
    '''Plot posterior predictions.'''

    probs = torch.mean(sampled_probs, axis=-1)  # compute probabilities from samples
    entropy = dist.Categorical(probs=probs).entropy()  # compute entropy from probabilities

    num_classes = probs.shape[-1]  # get number of classes

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=figsize)

    for idx, (ax1, ax2, ax3) in enumerate(axes):

        # image
        image = images[idx,0].numpy()
        ax1.imshow(image.clip(0, 1), cmap='gray')
        if labels is not None and names is not None:
            ax1.set_title(f'{names[labels[idx]]}')
        ax1.set(xticks=[], yticks=[], xlabel='', ylabel='')

        # violin plot
        # ax2.violinplot(sampled_probs[idx,:,:], positions=np.arange(num_classes))
        # ax2.set_title('$\pi(c|x,w)$, $w$ from $\pi(w|\mathcal{D})$')
        # ax2.set(xticks=np.arange(num_classes), ylim=(0, 1), xlabel='c')

        # histogram
        highest_ids = probs[idx].detach().cpu().numpy().argsort()[::-1][:3]
        for highest_idx in highest_ids:
            ax2.hist(
                sampled_probs[idx,highest_idx,:].detach().cpu().numpy(), bins=50,
                range=(0, 1), density=True, histtype='stepfilled', alpha=0.5
            )
        ax2.set_title('$\pi(c|x,w)$, $w$ from $\pi(w|\mathcal{D})$')
        ax2.set_xlim((0, 1))
        ax2.legend(['c={}'.format(c) for c in highest_ids], loc='upper center')
        ax2.grid(visible=True, which='both', color='lightgray', linestyle='-')
        ax2.set_axisbelow(True)

        # posterior predictive
        ax3.bar(np.arange(num_classes), probs[idx].detach().cpu().numpy())
        ax3.set_title('$\pi(c|x,\mathcal{D}) = \int \pi(c|x,w) \pi(w|\mathcal{D}) dw$')
        ax3.set(xticks=np.arange(num_classes), ylim=(0, 1), xlabel='c')
        ax3.text(0, 0.75, 'entropy: {:.2f}'.format(entropy[idx]), alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_entropy_histograms(
    norm_entropy,
    anom_entropy,
    figsize=(6, 4),
    range=(0, 2),
    bins=100,
    title='Predictions'
):
    '''Plot entropy histograms.'''

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        norm_entropy, bins=bins, range=range, density=True,
        histtype='stepfilled', alpha=0.7, label='in distribution'
    )

    ax.hist(
        anom_entropy, bins=bins, range=range, density=True,
        histtype='stepfilled', alpha=0.7, label='out of distribution'
    )

    ax.set(xlim=range, xlabel='entropy', ylabel='density')
    ax.set_title(title)
    ax.legend(loc='upper right')

    ax.grid(visible=True, which='both', color='lightgray', linestyle='-')
    ax.set_axisbelow(True)
    fig.tight_layout()

    return fig, ax

