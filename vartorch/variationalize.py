'''
Model variationalization.

Summary
-------
The core class 'VariationalClassification' implementes a scheme
for stochastic variational inference in classification problems.
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
from torchutils.tools import moving_average

class VariationalClassification(object):
    '''
    Variationalizing classification models.

    Summary
    -------
    This class supports stochastic variational inference by equipping classifier
    models including variational and normal layers with an MC simulator of the ELBO.
    This involves an expectation over the log-likelihood and a KL divergence term.
    The likelihood is defined based on a Categorical or a Bernoulli distribution.
    The KL divergence is automatically accumulated from the involved layers.
    While the likelihood term is estimated by means of MC sampling,
    the KL terms can be either analytical or simulated.

    Parameters
    ----------
    model : Pytorch module
        Logits-predicting model with variational layers.
    likelihood_type : {'Categorical', 'Bernoulli'}
        Determines the type of the likelihood function.
    device : PyTorch device
        Device the computations are performed on.

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
    with sampling (no_samples>1) and mean weight probabilites without (no_samples=1).
    The same behavior is found in 'predict_top' and 'test_acc', too.

    '''

    def __init__(self, model, likelihood_type='Categorical', device=None):
        self.model = model
        self.likelihood_type = likelihood_type
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = self.model.to(self.device)

    def sample(self, sample_mode=True):
        '''Set sampling mode of the model layers.'''
        for layer in self.model.modules():
            if hasattr(layer, 'sampling'):
                layer.sampling = sample_mode

    def train(self, train_mode=True):
        '''Set training mode of the model.'''
        self.model.train(train_mode)

    def ll(self, X, y):
        '''Compute the log-likelihood.'''
        y_logits = self.model(X) # log of unnormalized probabilities (scores)
        if self.likelihood_type == 'Categorical':
            # y_logprobs = nn.functional.log_softmax(y_logits, dim=1) # log-probabilities
            ll = dist.Categorical(logits=y_logits).log_prob(y).sum() # automatically normalizes
        elif self.likelihood_type == 'Bernoulli':
            # y_logprobs = nn.functional.logsigmoid(y_logits.squeeze()) # log-probabilities
            ll = dist.Bernoulli(logits=y_logits.squeeze()).log_prob(y.float()).sum() # automatically normalizes
        return ll

    def kl(self):
        '''Accumulate KL divergence from model layers.'''
        kl = torch.tensor(0.0, device=self.device)
        for layer in self.model.modules():
            if hasattr(layer, 'kl_acc'):
                kl += layer.kl_acc.to(self.device)
        return kl

    def elbo(self, X, y, no_samples=1, ll_weight=1, kl_weight=1):
        '''Simulate the ELBO by MC sampling.'''
        ll = torch.tensor(0.0, device=self.device)
        kl = torch.tensor(0.0, device=self.device)
        for idx in range(no_samples):
            ll += self.ll(X, y)
            kl += self.kl()
        ll /= no_samples
        kl /= no_samples
        elbo = ll * ll_weight - kl * kl_weight
        return elbo

    def loss(self, X, y, no_samples=1, ll_weight=1, kl_weight=1):
        '''Simulate the negative-ELBO loss.'''
        loss = -self.elbo(X, y, no_samples, ll_weight, kl_weight)
        return loss

    def __call__(self, X):
        '''Call model.'''
        y_logits = self.model(X)
        return y_logits

    def predict(self, X, no_samples=1):
        '''Predict logits.'''
        logits_list = []
        for idx in range(no_samples):
            logits = self.model(X)
            logits_list.append(logits)
        y_logits = torch.stack(logits_list, dim=-1).squeeze(dim=-1)
        return y_logits

    def predict_proba(self, X, no_samples=1):
        '''Predict probabilities (mean weight or posterior predictive).'''
        if no_samples == 1: # posterior mean weights
            self.sample(False)
            y_logits = self.predict(X)
            if self.likelihood_type == 'Categorical':
                y_probs = torch.softmax(y_logits, dim=1)
            elif self.likelihood_type == 'Bernoulli':
                y_probs = torch.sigmoid(y_logits)
        else: # posterior predictive distribution
            self.sample(True)
            sampled_logits = self.predict(X, no_samples)
            if self.likelihood_type == 'Categorical':
                sampled_probs = torch.softmax(sampled_logits, dim=1)
            elif self.likelihood_type == 'Bernoulli':
                sampled_probs = torch.sigmoid(sampled_logits)
            y_probs = torch.mean(sampled_probs, axis=-1)
        return y_probs

    def predict_top(self, X, no_samples=1, threshold=0.5):
        '''Predict top class and probability (mean weight or posterior predictive).'''
        y_probs = self.predict_proba(X, no_samples)
        if self.likelihood_type == 'Categorical':
            top_prob, top_class = torch.topk(y_probs, k=1, dim=1)
        elif self.likelihood_type == 'Bernoulli':
            top_class = (y_probs >= threshold).int()
            top_prob = torch.where(top_class==1, y_probs, 1-y_probs)
        return top_class, top_prob

    def compile_for_training(self, optimizer, train_loader, test_loader=None):
        '''Compile for training.'''
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def training(self, no_epochs, no_samples=1, log_interval=100, threshold=0.5, initial_test=True):
        '''Perform a number of training epochs.'''
        self.epoch = 0
        train_losses = []
        test_losses = []
        test_accs = []
        if initial_test:
            train_loss = self.test_loss(self.train_loader, no_samples, all_batches=False)
            test_loss = self.test_loss(self.test_loader, no_samples, all_batches=False)
            test_acc = self.test_acc(self.test_loader, no_samples=1, threshold=threshold)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            print('Started training: {}, test loss: {:.4e}, test acc.: {:.4f}' \
                  .format(self.epoch, test_loss, test_acc))
        for epoch_idx in range(no_epochs):
            train_loss = self.train_epoch(no_samples, log_interval)
            train_losses.append(train_loss)
            self.epoch += 1
            if self.test_loader is not None:
                test_loss = self.test_loss(no_samples=no_samples, all_batches=False)
                test_acc = self.test_acc(no_samples=1, threshold=threshold)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                print('Finished epoch: {}, test loss: {:.4e}, test acc.: {:.4f}' \
                      .format(self.epoch, test_loss, test_acc))
        history = {'no_epochs': no_epochs,
                   'train_loss': train_losses,
                   'test_loss': test_losses,
                   'test_acc': test_accs}
        return history

    def train_epoch(self, no_samples=1, log_interval=100):
        '''Perform a single training epoch.'''
        self.sample(True)
        self.train(True)
        batch_losses = []
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss(X_batch,
                             y_batch,
                             no_samples=no_samples,
                             ll_weight=len(self.train_loader.dataset) / X_batch.shape[0])
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.data.item()
            batch_losses.append(batch_loss)
            if len(batch_losses) < 3:
                running_loss = batch_loss
            else:
                running_loss = moving_average(batch_losses, window=3, mode='last')
            if log_interval is not None:
                if (batch_idx+1) % log_interval == 0 or (batch_idx+1) == len(self.train_loader):
                    print('Epoch: {} ({}/{}), batch loss: {:.4e}, running loss: {:.4e}' \
                          .format(self.epoch+1, batch_idx+1, len(self.train_loader), batch_loss, running_loss))
        return running_loss

    def test_loss(self, test_loader=None, no_samples=1, all_batches=False):
        '''Compute loss over a test set with one batch or all.'''
        if test_loader is None:
            test_loader = self.test_loader
        self.sample(True)
        self.train(False)
        with torch.no_grad():
            if all_batches: # all batches
                test_loss = 0.0
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    loss = self.loss(X_batch,
                                     y_batch,
                                     no_samples=no_samples,
                                     kl_weight=X_batch.shape[0] / len(test_loader.dataset))
                    test_loss += loss.data.item()
            else: # only one batch
                X_batch, y_batch = next(iter(test_loader))
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.loss(X_batch,
                                 y_batch,
                                 no_samples=no_samples,
                                 ll_weight=len(test_loader.dataset) / X_batch.shape[0])
                test_loss = loss.data.item()
        return test_loss

    def test_acc(self, test_loader=None, no_samples=1, no_epochs=1, threshold=0.5):
        '''Compute accuracy over a complete test set.'''
        if test_loader is None:
            test_loader = self.test_loader
        self.train(False)
        with torch.no_grad():
            no_total = 0
            no_correct = 0
            for epoch_idx in range(no_epochs):
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    no_total += X_batch.shape[0]
                    top_class, top_prob = self.predict_top(X_batch, no_samples, threshold)
                    is_correct = top_class.squeeze().int() == y_batch.squeeze().int()
                    no_correct += torch.sum(is_correct).data.item()
            test_acc = no_correct / no_total
        return test_acc

