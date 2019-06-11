# -*- coding: utf-8 -*-

import random

from basic_mlp.layers import Relu, Linear, Sigmoid
from basic_mlp.load_data import load_mnist_2d
from basic_mlp.loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from basic_mlp.network import Network
from basic_mlp.solve_net import train_net, test_net
from basic_mlp.utils import LOG_INFO

MAX = 1024
MIN = 1
GENE_LENGTH = 47
POPULATION_SIZE = 10
NG = 10
PC = 0.9
PM = 0.02


def binary2decimal(binary):
    """二进制转十进制"""
    return int(MIN + (int(binary, 2) * (MAX - MIN)) / (2 ** len(binary) - 1))


def active_function(flag, idx):
    if flag == '1':
        return Relu('relu{}'.format(idx))
    else:
        return Sigmoid('sigmoid{}'.format(idx))


def loss_function(flag):
    if flag == '1':
        return EuclideanLoss(name='EuclideanLoss')
    else:
        return SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss')


def print_neural_architecture(gene):
    number_of_layer = int(gene[:2], 2)
    print("gene: {}".format(gene))
    fc = []
    ac = []

    for idx, i in enumerate(range(3, 7)):
        ac.append(active_function(gene[i], idx + 1))

    for i in range(7, 47, 10):
        fc.append(binary2decimal(gene[i:i + 10]))

    print("fc0: 784, {}".format(fc[0]))
    print(ac[0])

    for i in range(number_of_layer):
        print("fc{}: {}, {}".format(i + 1, fc[i], fc[i + 1]))
        print("ac{}: {}".format(i + 1, ac[i + 1]))

    print("fc_end: {}, 10".format(fc[number_of_layer]))
    print(loss_function(gene[2]))


def decode(gene):
    # [number_of_layers:loss:ac0:ac1:ac2:ac3:fc0:fc1:fc2:fc3]
    number_of_layer = int(gene[:2], 2)

    fc = []
    ac = []

    for idx, i in enumerate(range(3, 7)):
        ac.append(active_function(gene[i], idx + 1))

    for i in range(7, 47, 10):
        fc.append(binary2decimal(gene[i:i + 10]))

    model = Network()
    model.add(Linear('fc0', 784, fc[0], 0.01))
    model.add(ac[0])

    for i in range(number_of_layer):
        model.add(Linear('fc{}'.format(i + 1), fc[i], fc[i + 1], 0.01))
        model.add(ac[i + 1])

    model.add(Linear('fc_end', fc[number_of_layer], 10, 0.01))
    model.loss = loss_function(gene[2])

    return model


def calculate_fitness(gene):
    print_neural_architecture(gene)

    model = decode(gene)

    config = {
        'learning_rate': 1e-1,
        'weight_decay': 1e-4,
        'momentum': 1e-4,
        'batch_size': 100,
        'max_epoch': 1,
        'disp_freq': 100,
        'test_epoch': 1
    }

    best_acc = 0

    for epoch in range(config['max_epoch']):
        # LOG_INFO('Training @ %d epoch...' % epoch)
        train_net(model, model.loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        _, acc = test_net(model, model.loss, test_data, test_label, config['batch_size'])
        if acc > best_acc:
            best_acc = acc
    return best_acc


def init_population():
    """初始化种群"""
    population = []
    for p in range(POPULATION_SIZE):
        tmp = ''
        for c in range(GENE_LENGTH):
            tmp += str(random.randint(0, 1))
        population.append(tmp)
    return population


def find_best_gene(population):
    best_gene = None
    best_value = -999
    total_fitness = 0
    for individual in population:
        value = calculate_fitness(individual)
        total_fitness += value
        if value > best_value:
            best_value = value
            best_gene = individual
    return best_gene, total_fitness


def crossover(population):
    for idx, individual in enumerate(population):
        if random.random() < PC:
            rand_spouse = random.randint(0, len(population) - 1)
            rand_point = random.randint(0, len(individual) - 1)

            new_parent_1 = individual[:rand_point] + population[rand_spouse][rand_point:]
            new_parent_2 = population[rand_spouse][:rand_point] + individual[rand_point:]

            population[idx] = new_parent_1
            population[rand_spouse] = new_parent_2
    return population


def mutation(population):
    new_population = []
    for individual in population:
        if random.random() < PM:
            rand_ = random.randint(0, len(individual) - 1)
            if individual[rand_] == '0':
                individual = individual[:rand_] + '1' + individual[rand_ + 1:]
            else:
                individual = individual[:rand_] + '0' + individual[rand_ + 1:]
        new_population.append(individual)
    return new_population


def binary_search(_list, item):
    """假装是二分查找"""
    if item <= _list[0]:
        return 0
    for idx in range(len(_list) - 1):
        if _list[idx] < item <= _list[idx + 1]:
            return idx + 1


def selection(population, total_fitness):
    """旋轮法选择"""
    p_value = []
    new_population = []
    for individual in population:
        fitness = calculate_fitness(individual)
        p_value.append(fitness / total_fitness)
    cdf = cumulative_distribution_function(p_value)
    for _iter in range(POPULATION_SIZE):
        probability = random.random()
        new_population.append(population[binary_search(cdf, probability)])
    return new_population


def cumulative_distribution_function(p_value):
    """Cumulative Distribution Function"""
    cdf = []
    for idx in range(len(p_value)):
        cdf.append(sum(p_value[:idx + 1]))
    return cdf


def main():
    # 产生初始种群
    population = init_population()
    best_gene = '0' * GENE_LENGTH
    best_gene_history = '0' * GENE_LENGTH
    _iter = 0
    for _iter in range(NG):
        LOG_INFO('迭代次数: {}, 本轮最优值: {}, 最优基因代码: {}'.format(_iter, calculate_fitness(best_gene), best_gene))
        # 计算适值函数
        best_gene, total_fitness = find_best_gene(population)
        # 选择
        population = selection(population, total_fitness)
        # 遗传运算
        population = crossover(population)
        population = mutation(population)

        if calculate_fitness(best_gene) > calculate_fitness(best_gene_history):
            best_gene_history = best_gene

    LOG_INFO('迭代次数: {}'.format(_iter))
    LOG_INFO("计算结果: {}".format(calculate_fitness(best_gene_history)))
    LOG_INFO("基因片段: {}".format(best_gene_history))


if __name__ == '__main__':
    train_data, test_data, train_label, test_label = load_mnist_2d('data')
    main()
    # print_neural_architecture('01011111111100000101001111011011000100010000111')
