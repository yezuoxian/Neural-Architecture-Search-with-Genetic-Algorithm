from layers import Relu, Linear
from load_data import load_mnist_2d
from loss import EuclideanLoss
from network import Network
from solve_net import train_net, test_net
from utils import LOG_INFO

if __name__ == '__main__':
    train_data, test_data, train_label, test_label = load_mnist_2d('../data')

    # Your model definition here
    # You should explore different model architecture
    # model = Network()
    # model.add(Linear('fc1', 784, 993, 0.01))
    # model.add(Relu('relu1'))
    # model.add(Linear('fc2', 993, 671, 0.01))
    # model.add(Relu('relu2'))
    # model.add(Linear('fc3', 671, 10, 0.01))
    # model.loss = SoftmaxCrossEntropyLoss(name='loss')

    # model = Network()
    # model.add(Linear('fc1', 784, 815, 0.01))
    # model.add(Relu('relu1'))
    # model.add(Linear('fc3', 815, 10, 0.01))
    # model.loss = EuclideanLoss(name='loss')

    model = Network()
    model.add(Linear('fc1', 784, 783, 0.01))
    model.add(Relu('relu1'))
    model.add(Linear('fc3', 783, 10, 0.01))
    model.loss = EuclideanLoss(name='loss')

    # Training configuration
    # You should adjust these hyperparameters
    # NOTE: one iteration means model forward-backwards one batch of samples.
    #       one epoch means model has gone through all the training samples.
    #       'disp_freq' denotes number of iterations in one epoch to display information.

    config = {
        'learning_rate': 1e-1,
        'weight_decay': 1e-4,
        'momentum': 1e-4,
        'batch_size': 100,
        'max_epoch': 200,
        'disp_freq': 100,
        'test_epoch': 1
    }

    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % epoch)
        train_net(model, model.loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % epoch)
            test_net(model, model.loss, test_data, test_label, config['batch_size'])
