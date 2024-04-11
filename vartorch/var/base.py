'''
Variational classifier.

Summary
-------
The core class 'VarClassifier' implements a scheme for
stochastic variational inference in classification problems.
It turns models with variational and other layers into Bayesian classifiers.
While the likelihood determines whether the problem is binary or multi-class,
the model layers determine the unknown weights and their treatment.

A posterior over the weights of the variational layers is computed by
maximizing the ELBO w.r.t. the parameters of the variational distribution.
Other non-Bayesian unknown parameters of the prior and the likelihood,
such as the weights of non-variational layers, can also be learned this way.

'''

import torch
import torch.distributions as dist
from lightning.pytorch import LightningModule
from torchmetrics.classification import Accuracy


class VarClassifier(LightningModule):
    '''
    Variational classifier base class.

    Summary
    -------
    This class supports stochastic variational inference by equipping classifier
    models including variational and normal layers with an MC simulator of the ELBO.
    This involves an expectation over the log-likelihood and a KL divergence term.

    The likelihood is defined based on a categorical or a Bernoulli distribution.
    The KL divergence is automatically accumulated from the involved layers.
    While the likelihood term is estimated by means of MC sampling,
    the KL terms can be either analytical or simulated.

    Notes
    -----
    Some care regarding the sampling behavior has to be taken.
    When sampling is turned on, the model predicts with weights
    that are randomly drawn from the variational distribution.
    When turned off, however, the weights assume their mean value.
    The 'predict'-method therefore has a context-dependent behavior.

    Sampling has to be manually turned on (or off) for the loss simulation.
    Some high-level methods automatically use the appropriate sampling mode.
    An example is 'predict_proba' which computes posterior pred. probabilities with
    sampling (num_samples>1) and mean weight probabilities without (num_samples=1).

    Parameters
    ----------
    model : Pytorch module
        Logits-predicting model with variational layers.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'Categorical'}
        Likelihood function type.
    num_classes : int
        Number of classes.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 model,
                 num_samples=1,
                 likelihood_type='Categorical',
                 num_classes=None,
                 lr=1e-04):

        super().__init__()

        # set model
        self.model = model

        # set number of MC samples
        self.num_samples = abs(int(num_samples))

        # set likelihood type
        if likelihood_type in ('Bernoulli', 'Categorical'):
            self.likelihood_type = likelihood_type
        else:
            raise ValueError(f'Unknown likelihood type: {likelihood_type}')

        # set initial learning rate
        self.lr = abs(lr)

        # set initial sampling mode
        self.sample(True)

        # store hyperparams
        self.save_hyperparameters(
            ignore='model',
            logger=True
        )

        # create accuracy metrics
        if self.likelihood_type == 'Bernoulli':
            self.train_acc = Accuracy(task='binary')
            self.val_acc = Accuracy(task='binary')
            self.test_acc = Accuracy(task='binary')

        elif num_classes is not None:
            if num_classes < 2:
                raise ValueError('Number of classes should be at least two')

            self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
            self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
            self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    @property
    def sampling(self):

        # get sampling mode per layer
        per_layer = []
        for layer in self.model.modules():
            if hasattr(layer, 'sampling'):
                per_layer.append(layer.sampling)

        # determine global sampling mode
        if len(per_layer) == 0:
            raise RuntimeError('No layers with sampling mode found')

        elif len(per_layer) == 1:
            sampling = per_layer[0]

        else:
            first_mode = per_layer[0]
            all_same = [(s == first_mode) for s in per_layer[1:]]

            if all_same:
                sampling = first_mode
            else:
                raise RuntimeError('Inconsistent sampling modes encountered')

        return sampling

    @sampling.setter
    def sampling(self, sample_mode):

        # set sampling mode for all model layers
        for layer in self.model.modules():
            if hasattr(layer, 'sampling'):
                layer.sampling = sample_mode

    def sample(self, sample_mode=True):
        '''Set sampling mode.'''
        self.sampling = sample_mode
        return self

    def train(self, train_mode=True):
        '''Set training mode.'''

        # set module training mode
        super().train(train_mode)

        # turn on sampling for training
        if train_mode:
            self.sample(True)

        return self

    def forward(self, x):
        '''Run model.'''
        logits = self.model(x)
        return logits

    def predict(self, x, num_samples=1):
        '''Predict logits.'''

        # loop over samples
        logits_samples = []
        for _ in range(num_samples):
            logits_sample = self(x)
            logits_samples.append(logits_sample)

        # stack samples
        logits_samples = torch.stack(logits_samples, dim=-1).squeeze(dim=-1)

        return logits_samples

    def probs_from_logits(self, logits, average=True):
        '''Compute (average) probabilities from (samples of) logits.'''

        if logits.ndim not in (2, 3):
            raise ValueError(f'Invalid input shape: {logits.shape}')

        # calculate probabilities
        if self.likelihood_type == 'Bernoulli':
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)

        # average over samples if needed
        if average and (probs.ndim == 3):
            probs = torch.mean(probs, dim=-1)

        return probs

    def predict_proba(self, x, num_samples=1):
        '''Predict probabilities (mean weight or posterior predictive).'''

        # predict with posterior mean weights
        if num_samples == 1:

            # turn off sampling
            self.sample(False)

            # predict logits
            logits = self.predict(x, num_samples=1)

            # calculate probabilities
            probs = self.probs_from_logits(logits)

        # compute posterior predictive distribution
        else:

            # turn on sampling
            self.sample(True)

            # predict logits
            logits_samples = self.predict(x, num_samples=num_samples)

            # calculate probabilities
            probs = self.probs_from_logits(logits_samples)

        return probs

    def predict_top(self,
                    x,
                    num_samples=1,
                    threshold=0.5):
        '''Predict top class and probability (mean weight or posterior predictive).'''

        # predict probabilities
        probs = self.predict_proba(x, num_samples=num_samples)

        # get top class and its probability
        if self.likelihood_type == 'Bernoulli':
            top_class = (probs >= threshold).int()
            top_prob = torch.where(top_class==1, probs, 1 - probs)
        else:
            top_prob, top_class = torch.topk(probs, k=1, dim=1)

        return top_class, top_prob

    def kl(self):
        '''Accumulate KL divergence from model layers.'''

        kl = 0.0

        # accumulate KL div. from appropriate layers
        for layer in self.model.modules():
            if hasattr(layer, 'kl_acc'):
                kl = kl + layer.kl_acc.to(self.device)

        return kl

    def ll(self, x, y, return_preds=False):
        '''Compute the log-likelihood.'''

        # predict logits
        logits = self(x)

        # compute log-likelihood
        if self.likelihood_type == 'Bernoulli':
            ll = dist.Bernoulli(logits=logits.squeeze()).log_prob(y.float()).sum()
        else:
            ll = dist.Categorical(logits=logits).log_prob(y).sum()

        if return_preds:
            return ll, logits
        else:
            return ll

    def elbo(self,
             x,
             y,
             num_samples=1,
             ll_weight=1.0,
             kl_weight=1.0,
             return_preds=False):
        '''Simulate the ELBO by MC sampling.'''

        ll = 0.0
        kl = 0.0

        if return_preds:
            logits_samples = []

        # loop over samples
        for _ in range(num_samples):

            out = self.ll(x, y, return_preds=return_preds) # during the LL computation also the KL terms are calculated

            if return_preds:
                ll_sample, logits_sample = out
                logits_samples.append(logits_sample)
            else:
                ll_sample = out

            ll = ll + ll_sample
            kl = kl + self.kl() # the KL terms can be aggregated after the LL computation

        # average over samples
        ll = ll / num_samples
        kl = kl / num_samples

        elbo = ll * ll_weight - kl * kl_weight

        if return_preds:
            logits_samples = torch.stack(logits_samples, dim=-1).squeeze(dim=-1)
            probs = self.probs_from_logits(logits_samples)
            return elbo, probs
        else:
            return elbo

    def loss(self,
             x,
             y,
             num_samples=1,
             total_size=None,
             reweight_ll=True,
             return_preds=False):
        '''Simulate the negative-ELBO loss.'''

        # get weighting factors
        ll_weight = 1.0
        kl_weight = 1.0

        if total_size is not None:
            batch_size = x.shape[0]

            # up-weight LL in order to obtain a batch estimate for the full dataset (use when averaging)
            if reweight_ll:
                ll_weight = total_size / batch_size

            # down-weight KLD so as to compensate for when it is summed over multiple times (use when summing)
            else:
                kl_weight = batch_size / total_size

        # compute the ELBO
        out = self.elbo(
            x,
            y,
            num_samples=num_samples,
            ll_weight=ll_weight,
            kl_weight=kl_weight,
            return_preds=return_preds
        )

        if return_preds:
            elbo, probs = out
        else:
            elbo = out

        # get the loss (negative ELBO)
        loss = -elbo

        if return_preds:
            return loss, probs
        else:
            return loss

    @staticmethod
    def _get_batch(batch):
        '''Get batch features and labels.'''

        if isinstance(batch, (tuple, list)):
            x_batch = batch[0]
            y_batch = batch[1]

        elif isinstance(batch, dict):
            x_batch = batch['features']
            y_batch = batch['labels']

        else:
            raise TypeError(f'Invalid batch type: {type(batch)}')

        return x_batch, y_batch

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)

        loss, probs = self.loss(
            x_batch,
            y_batch,
            num_samples=self.num_samples,
            total_size=len(self.trainer.train_dataloader.dataset),
            reweight_ll=True, # up-weight LL so as to estimate the full dataset loss (batches can be better compared)
            return_preds=True
        )

        _ = self.train_acc(probs.squeeze(), y_batch)

        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default
        self.log('train_acc', self.train_acc) # the same applies to torchmetrics.Metric objects
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)

        loss, probs = self.loss(
            x_batch,
            y_batch,
            num_samples=self.num_samples,
            total_size=len(self.trainer.val_dataloaders.dataset),
            reweight_ll=True, # up-weight LL so as to estimate the full dataset loss (can be averaged)
            return_preds=True
        )

        _ = self.val_acc(probs.squeeze(), y_batch)

        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation
        self.log('val_acc', self.val_acc) # the batch size is considered when logging torchmetrics.Metric objects
        return loss

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)

        loss, probs = self.loss(
            x_batch,
            y_batch,
            num_samples=1, # use a single sample (mean value), since sampling is off while testing
            total_size=len(self.trainer.test_dataloaders.dataset),
            reweight_ll=True, # up-weight LL so as to estimate the full dataset loss (can be averaged)
            return_preds=True
        )

        _ = self.test_acc(probs.squeeze(), y_batch)

        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing
        self.log('test_acc', self.test_acc) # the batch size is considered when logging torchmetrics.Metric objects
        return loss

    def on_train_epoch_start(self):
        self.sample(True) # turn on sampling for training

    def on_validation_epoch_start(self):
        self.sample(True) # turn on sampling for validation

    def on_test_epoch_start(self):
        self.sample(False) # turn off sampling for testing

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # create reduce-on-plateau schedule
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer
        # )

        # create cosine annealing schedule
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs
        )

        # lr_config = {
        #     'scheduler': lr_scheduler, # LR scheduler
        #     'interval': 'epoch', # time unit
        #     'frequency': 1 # update frequency
        # }

        return [optimizer], [lr_scheduler]

