import abc
import torch
import torch.nn as nn
import torchvision

class BaseModule(abc.ABC, nn.Module):
    """
      Base class for ensemble methods.

      WARNING: This class cannot be used directly.
      Use the derived classes instead.
    """

    def __init__(self,
                 estimators,
                 n_estimators,
                 output_dim,
                 lr,
                 weight_decay,
                 epochs,
                 cuda=True,
                 log_interval=100,
                 n_jobs=1):
        """
        Parameters
        ----------
        estimator : torch.nn.Module
            The class of base estimator inherited from `torch.nn.Module`.
        n_estimators : int
            The number of base estimators in the ensemble model.
        output_dim : int
            The output dimension of the model. For instance, for multi-class
            classification problem with K classes, it is set to `K`. For
            univariate regression problem, it is set to `1`.
        lr : float
            The learning rate of the parameter optimizer.
        weight_decay : float
            The weight decay of the parameter optimizer.
        epochs : int
            The number of training epochs.
        cuda : bool, default=True
            When set to `True`, use GPU to train and evaluate the model. When
            set to `False`, the model is trained and evaluated using CPU.
        log_interval : int, default=100
            The number of batches to wait before printing the training status,
            including information on the current batch, current epoch, current
            training loss, and many more.
        n_jobs : int, default=1
            The number of workers for training the ensemble model. This
            argument is used for parallel ensemble methods such as voting and
            bagging. Setting it to an integer larger than `1` enables many base
            estimators to be jointly trained. However, training many base
            estimators at the same time may run out of the memory.

        Attributes
        ----------
        estimators_ : nn.ModuleList
            A container that stores all base estimators.

        """
        super(BaseModule, self).__init__()
        self.estimator = ""

        self.estimators_ = nn.ModuleList()
        # in this version we already have initialized estimators
        for estimator in estimators:
            self.estimators_.append(estimator)
        self.n_estimators = n_estimators
        self.output_dim = output_dim

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.log_interval = log_interval
        self.n_jobs = n_jobs
        self.device = torch.device('cuda' if cuda else 'cpu')
        for estimator in self.estimators_:
            estimator = estimator.to(device=self.device)

        # A global optimizer
        params = list()
        for estimator in self.estimators_:
            if isinstance(estimator.model,torchvision.models.resnet.ResNet):
                params += list(estimator.model.model.layer4.parameters())
                params += list(estimator.model.model.fc.parameters())
            elif isinstance(estimator.model,torchvision.models.inception.Inception3):
                params += list(estimator.model.mixed7c.parameters())
                params += list(estimator.model.fc.parameters(()))
            else:
                params += list(estimator.parameters())

        self.optimizer = torch.optim.Adam(params,
                                          lr=lr, weight_decay=weight_decay)

    def __str__(self):
        msg = '==============================\n'
        name = ""
        for i in range(len(self.estimators_)):
            if i < len(self.estimators_)-1:
                name += self.estimators_[i].__name__() + ", "
            else:
                name += self.estimators_[i].__name__()

        msg += '{:<20}: {}\n'.format('Estimators',
                                     name)
        msg += '{:<20}: {}\n'.format('n_estimators', self.n_estimators)
        msg += '{:<20}: {:.5f}\n'.format('Learning Rate', self.lr)
        msg += '{:<20}: {:.5f}\n'.format('Weight Decay', self.weight_decay)
        msg += '{:<20}: {}\n'.format('n_epochs', self.epochs)
        msg += '==============================\n'

        return msg

    def __repr__(self):
        return self.__str__()

    def _validate_parameters(self):

        if not self.n_estimators > 0:
            msg = ('The number of base estimators = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.n_estimators))

        if not self.output_dim > 0:
            msg = 'The output dimension = {} should be strictly positive.'
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

    @abc.abstractmethod
    def forward(self, X):
        """
        Implementation on the data forwarding in the ensemble model. Notice
        that the input `X` should be a data batch instead of a standalone
        data loader that contains all data batches.
        """

    @abc.abstractmethod
    def fit(self, train_loader):
        """
        Implementation on the training stage of the ensemble model.
        """

    @abc.abstractmethod
    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of the ensemble model.
        """
