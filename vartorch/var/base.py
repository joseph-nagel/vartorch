'''Variational classifier.'''

import torch
import torch.distributions as dist
from lightning.pytorch import LightningModule


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

    Some high-level methods automatically use the appropriate mode, for instance,
    'train_epoch' or 'test_loss' have to use sampling for the loss simulation.
    Another example is 'predict_proba' that computes posterior predictive probabilities
    with sampling (num_samples>1) and mean weight probabilities without (num_samples=1).
    The same behavior is found in 'predict_top' and 'test_acc', too.

    Parameters
    ----------
    model : Pytorch module
        Logits-predicting model with variational layers.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'Categorical'}
        Likelihood function type.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 model,
                 num_samples=1,
                 likelihood_type='Categorical',
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

    @property
    def sampling(self):

        # get sampling mode per layer
        per_layer = [l.sampling for l in self.model.modules()]

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
        y_logits = self.model(x)
        return y_logits

    def predict(self, x, num_samples=1):
        '''Predict logits.'''

        # loop over samples
        logits_list = []
        for _ in range(num_samples):
            logits = self(x)
            logits_list.append(logits)

        # stack samples
        y_logits = torch.stack(logits_list, dim=-1).squeeze(dim=-1)

        return y_logits

    def predict_proba(self, x, num_samples=1):
        '''Predict probabilities (mean weight or posterior predictive).'''

        # predict with posterior mean weights
        if num_samples == 1:

            # turn off sampling
            self.sample(False)

            # predict logits
            y_logits = self.predict(x, num_samples=1)

            # calculate probabilities
            if self.likelihood_type == 'Bernoulli':
                y_probs = torch.sigmoid(y_logits)
            else:
                y_probs = torch.softmax(y_logits, dim=1)

        # compute posterior predictive distribution
        else:

            # turn on sampling
            self.sample(True)

            # predict logits
            sampled_logits = self.predict(x, num_samples=num_samples)

            # calculate probabilities
            if self.likelihood_type == 'Bernoulli':
                sampled_probs = torch.sigmoid(sampled_logits)
            else:
                sampled_probs = torch.softmax(sampled_logits, dim=1)

            # average over samples
            y_probs = torch.mean(sampled_probs, dim=-1)

        return y_probs

    def predict_top(self,
                    x,
                    num_samples=1,
                    threshold=0.5):
        '''Predict top class and probability (mean weight or posterior predictive).'''

        # predict probabilities
        y_probs = self.predict_proba(x, num_samples=num_samples)

        # get top class and its probability
        if self.likelihood_type == 'Bernoulli':
            top_class = (y_probs >= threshold).int()
            top_prob = torch.where(top_class==1, y_probs, 1 - y_probs)
        else:
            top_prob, top_class = torch.topk(y_probs, k=1, dim=1)

        return top_class, top_prob

    def kl(self):
        '''Accumulate KL divergence from model layers.'''

        kl = 0.0

        # accumulate KL div. from appropriate layers
        for layer in self.model.modules():
            if hasattr(layer, 'kl_acc'):
                kl = kl + layer.kl_acc.to(self.device)

        return kl

    def ll(self, x, y):
        '''Compute the log-likelihood.'''

        # predict logits
        y_logits = self(x)

        # compute log-likelihood
        if self.likelihood_type == 'Bernoulli':
            ll = dist.Bernoulli(logits=y_logits.squeeze()).log_prob(y.float()).sum()
        else:
            ll = dist.Categorical(logits=y_logits).log_prob(y).sum()

        return ll

    def elbo(self,
             x,
             y,
             num_samples=1,
             ll_weight=1.0,
             kl_weight=1.0):
        '''Simulate the ELBO by MC sampling.'''

        ll = 0.0
        kl = 0.0

        # loop over samples
        for _ in range(num_samples):
            ll = ll + self.ll(x, y)
            kl = kl + self.kl()

        # avergage over samples
        ll = ll / num_samples
        kl = kl / num_samples

        elbo = ll * ll_weight - kl * kl_weight

        return elbo

    def loss(self,
             x,
             y,
             num_samples=1,
             total_size=None,
             reweight_ll=True):
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
        elbo = self.elbo(
            x,
            y,
            num_samples=num_samples,
            ll_weight=ll_weight,
            kl_weight=kl_weight
        )

        # get the loss (negative ELBO)
        loss = -elbo

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

        loss = self.loss(
            x_batch,
            y_batch,
            num_samples=self.num_samples,
            total_size=len(self.trainer.train_dataloader.dataset),
            reweight_ll=True # up-weight LL so as to estimate the full dataset loss (batches can be better compared)
        )

        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)

        loss = self.loss(
            x_batch,
            y_batch,
            num_samples=self.num_samples,
            total_size=len(self.trainer.val_dataloaders.dataset),
            reweight_ll=True # up-weight LL so as to estimate the full dataset loss (can be averaged)
        )

        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = self._get_batch(batch)

        loss = self.loss(
            x_batch,
            y_batch,
            num_samples=self.num_samples,
            total_size=len(self.trainer.test_dataloaders.dataset),
            reweight_ll=True # up-weight LL so as to estimate the full dataset loss (can be averaged)
        )

        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

