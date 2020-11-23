"""
  Gradient boosting is a classic sequential ensemble method. At each iteration,
  the learning target of a new base estimator is to fit the pseudo residual
  computed based on the ground truth and the output from base estimators
  fitted before, using ordinary least square.
"""
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ._base import BaseModule


class BaseGradientBossting(BaseModule):

    def __init__(self, estimators, n_estimators, output_dim,
                 lr, weight_decay, epochs,
                 shrinkage_rate=1., cuda=True, log_interval=100):
        super(BaseModule, self).__init__()
        self.estimator = "" # afraid of breaking something without it...
        self.estimators_ = nn.ModuleList()
        # in this version we already have initialized estimators
        for estimator in estimators:
            self.estimators_.append(estimator)
        self.n_estimators = n_estimators
        self.output_dim = output_dim

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.shrinkage_rate = shrinkage_rate

        self.log_interval = log_interval
    def _validate_parameters(self):

        if not self.n_estimators > 0:
            msg = ('The number of base estimators = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.n_estimators))

        if not self.output_dim > 0:
            msg = 'The output dimension = {} should not be strictly positive.'
            raise ValueError(msg.format(self.output_dim))

        if not self.lr > 0:
            msg = ('The learning rate of optimizer = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.lr))

        if not self.weight_decay >= 0:
            msg = 'The weight decay of parameters = {} should not be negative.'
            raise ValueError(msg.format(self.weight_decay))

        if not self.epochs > 0:
            msg = ('The number of training epochs = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.epochs))

        if not 0 < self.shrinkage_rate <= 1:
            msg = ('The shrinkage rate should be in the range (0, 1], but got'
                   ' {} instead.')
            raise ValueError(msg.format(self.shrinkage_rate))

    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)

        # The output of `GradientBoostingRegressor` is the summation of output
        # from all base estimators, with each of them multipled by the
        # shrinkage rate.
        for estimator in self.estimators_:
            estimator = estimator.to(self.device)
            y_pred += self.shrinkage_rate * estimator(X)

        return y_pred

    def fit(self, train_loader, val_loader):
        cuda = torch.cuda.is_available()
        self.train()
        self._validate_parameters()
        criterion = nn.MSELoss(reduction='sum')

        # Base estimators are fitted sequentially in gradient boosting
        for est_idx, estimator in enumerate(self.estimators_):

            # Initialize an independent optimizer for each base estimator to
            # avoid unexpected dependencies.
            params = list()
            if isinstance(estimator.model, torchvision.models.resnet.ResNet):
                params += list(estimator.model.model.layer4.parameters())
                params += list(estimator.model.model.fc.parameters())
            elif isinstance(estimator.model, torchvision.models.inception.Inception3):
                params += list(estimator.model.Mixed_7c.parameters())
                params += list(estimator.model.fc.parameters(()))
            else:
                params += list(estimator.parameters())
            learner_optimizer = torch.optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay)

            for epoch in range(self.epochs):
                for batch_idx, (X_train, y_train) in enumerate(train_loader):

                    X_train, y_train = (X_train.to(self.device),
                                        y_train.to(self.device))

                    # Learning target of the estimator with index `est_idx`
                    y_residual = self._pseudo_residual(X_train, y_train,
                                                       est_idx)

                    output = estimator(X_train)
                    loss = criterion(output, y_residual)

                    learner_optimizer.zero_grad()
                    loss.backward()
                    learner_optimizer.step()

                    # Print training status
                    if batch_idx % self.log_interval == 0:
                        msg = ('Estimator: {:03d} | Epoch: {:03d} | Batch:'
                               ' {:03d} | RegLoss: {:.5f}')
                        print(msg.format(est_idx, epoch, batch_idx, loss))
                # this will fail on regressor ?
                if epoch % 5 == 0:
                    self.predict(val_loader)

class GradientBoostingClassifier(BaseGradientBossting):

    def _onehot_coding(self, y):
        """ Convert the class label to a one-hot encoded vector. """

        y = y.view(-1)
        y_onehot = torch.FloatTensor(
            y.size()[0], self.output_dim).to(self.device)
        y_onehot.data.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)

        return y_onehot

    # TODO: Store the output of *fitted* base estimators to avoid repeated data
    # forwarding. Since samples in the data loader can be shuffled, it requires
    # the index of each sample in the original dataset to be kept in memory.

    # TODO: Implement second order learning target, which is used in existing
    # decision tree based GBDT systems like XGBoost and LightGBM. However, this
    # can be problematic using estimators like neural network, as the second
    # order target can be orders of magnitude larger than the first order
    # target, making it hard to conduct ERM on the squared loss using gradient
    # descent based optimization strategies.
    def _pseudo_residual(self, X, y, est_idx):
        y_onehot = self._onehot_coding(y)
        output = torch.zeros_like(y_onehot).to(self.device)

        if est_idx == 0:
            # Before training the first estimator, we assume that the GBM model
            # always returns `0` for any input (i.e., null output).
            return y_onehot - F.softmax(output, dim=1)
        else:
            for idx in range(est_idx):
                output += self.shrinkage_rate * self.estimators_[idx](X)

            return y_onehot - F.softmax(output, dim=1)

    def predict(self, test_loader):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.estimators_.to(self.device)
        self.eval()
        correct = 0.

        with torch.no_grad():
            for batch_idx, (X_test, y_test) in enumerate(test_loader):

                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                output = F.softmax(self.forward(X_test), dim=1)
                y_pred = output.data.max(1)[1]
                correct += y_pred.eq(y_test.view(-1).data).sum()

            accuracy = 100. * float(correct) / len(test_loader.dataset)
        print("Test accuracy {} %".format(accuracy))

        return accuracy

    def blind_predict(self,test_loader):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.estimators_.to(self.device)
        self.eval()
        res = dict()
        with torch.no_grad():
            for images, name in test_loader:
                name = name[0]
                images = [img.to(self.device) for img in images]
                if len(images) == 2:
                    output1 = F.softmax(self.forward(images[0]),dim=1)
                    output2 = F.softmax(self.forward(images[1]),dim=1)
                    output = 0.5 * output1 + 0.5 * output2
                else:
                    output1 = F.softmax(self.forward(images[0]), dim=1)
                    output2 = F.softmax(self.forward(images[1]), dim=1)
                    output3 = F.softmax(self.forward(images[2]), dim=1)
                    output4 = F.softmax(self.forward(images[3]), dim=1)
                    output = 0.2 * output1 + 0.2 * output2 + 0.3 * output3 + 0.3 * output4
                pred = output.data.max(1)[1]
                res[name] = pred
        return res


    def val_predict(self, val_loader):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.estimators_.to(self.device)
        self.eval()
        correct = 0
        with torch.no_grad():
            for images, label in val_loader:
                images = [img.to(self.device) for img in images]
                if len(images) == 2:
                    output1 = F.softmax(self.forward(images[0]), dim=1)
                    output2 = F.softmax(self.forward(images[1]), dim=1)
                    output = 0.5 * output1 + 0.5 * output2
                else:
                    output1 = F.softmax(self.forward(images[0]), dim=1)
                    output2 = F.softmax(self.forward(images[1]), dim=1)
                    output3 = F.softmax(self.forward(images[2]), dim=1)
                    output4 = F.softmax(self.forward(images[4]), dim=1)
                    output = 0.2 * output1 + 0.2 * output2 + 0.3 * output3 + 0.3 * output4
                pred = output.data.max(1)[1]
                correct += label.eq(label.view(-1).data).sum()
            accuracy = 100. * float(correct) / len(val_loader.dataset)

        print("Test accuracy {} %".format(accuracy))



class GradientBoostingRegressor(BaseGradientBossting):

    def _pseudo_residual(self, X, y, est_idx):
        output = torch.zeros_like(y).to(self.device)

        if est_idx == 0:
            # Before training the first estimator, we assume that the GBM model
            # always returns `0` for any input (i.e., null output).
            return y
        else:
            for idx in range(est_idx):
                output += self.shrinkage_rate * self.estimators_[idx](X)

            return y - output

    def predict(self, test_loader):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.estimators_.to(self.device)
        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)

            mse += criterion(output, y_test)

        return mse / len(test_loader)

