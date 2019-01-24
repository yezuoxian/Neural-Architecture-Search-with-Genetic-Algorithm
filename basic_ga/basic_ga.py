# -*- coding: utf-8 -*-
import random

import numpy as np

MAX = 100
MIN = -100
GENE_LENGTH = 40
POPULATION_SIZE = 1000
NG = 200
PC = 0.9
PM = 0.02


def j_d_schaffer(x1, x2):
    up = np.square(np.sin(np.sqrt(np.square(x1) + np.square(x2)))) - 0.5
    down = np.square(1 + 0.001 * (np.square(x1) + np.square(x2)))
    return 4.5 - up / down


def binary2decimal(binary):
    '''二进制转十进制'''
    return MIN + (int(binary, 2) * (MAX - MIN)) / (2 ** len(binary) - 1)


def decode(gene):
    '''将基因片段转化成实数，基因构成为[x1:x2]'''
    x_length = int(GENE_LENGTH / 2)
    x1_gene = gene[:x_length]
    x2_gene = gene[x_length:]
    x1 = binary2decimal(x1_gene)
    x2 = binary2decimal(x2_gene)
    return x1, x2


def calculate_fitness(gene):
    '''计算适应性评分，即目标函数值'''
    x1, x2 = decode(gene)
    return j_d_schaffer(x1, x2)


def init_population():
    '''初始化种群'''
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
        if (random.random() < PC):
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
        if (random.random() < PM):
            rand_ = random.randint(0, len(individual) - 1)
            if individual[rand_] == '0':
                individual = individual[:rand_] + '1' + individual[rand_ + 1:]
            else:
                individual = individual[:rand_] + '0' + individual[rand_ + 1:]
        new_population.append(individual)
    return new_population


def binary_search(list, item):
    '''假装是二分查找'''
    if item <= list[0]:
        return 0
    for idx in range(len(list) - 1):
        if list[idx] < item and list[idx + 1] >= item:
            return idx + 1


def selection(population, total_fitness):
    '''旋轮法选择'''
    p_value = []
    new_population = []
    for individual in population:
        fitness = calculate_fitness(individual)
        p_value.append(fitness / total_fitness)
    cdf = cumulative_distribution_function(p_value)
    for iter in range(POPULATION_SIZE):
        probability = random.random()
        new_population.append(population[binary_search(cdf, probability)])
    return new_population


def cumulative_distribution_function(p_value):
    '''cumulative_Distribution_Function'''
    cdf = []
    for idx in range(len(p_value)):
        cdf.append(sum(p_value[:idx + 1]))
    return cdf


def main():
    # 产生初始种群
    population = init_population()
    best_gene = '0' * GENE_LENGTH
    best_gene_history = '0' * GENE_LENGTH
    for iter in range(NG):
        print('迭代次数:{}, 本轮最优值:{}'.format(iter, calculate_fitness(best_gene)))
        # 计算适值函数
        best_gene, total_fitness = find_best_gene(population)
        # 选择
        population = selection(population, total_fitness)
        # 遗传运算
        population = crossover(population)
        population = mutation(population)

        if calculate_fitness(best_gene) > calculate_fitness(best_gene_history):
            best_gene_history = best_gene

    print('迭代次数:{}'.format(iter))
    print("计算结果: {}".format(calculate_fitness(best_gene_history)))
    print("基因片段: {}".format(best_gene_history))
    print("x1={}, x2={}".format(decode(best_gene_history)[0], decode(best_gene_history)[1]))


if __name__ == '__main__':
    main()
