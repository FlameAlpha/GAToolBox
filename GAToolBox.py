import os
import threading
import time as tm
import numpy as np
import pandas as pd
from enum import Enum
from time import sleep
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame


"""
集合运算
"""


def bin_union_list(factor1, factor2):
    """ 求并集 """
    factor = []
    for i, (g1, g2) in enumerate(zip(factor1, factor2)):
        factor.append(g1 | g2)
    return factor


def bin_intersection_list(factor1, factor2):
    """ 求交集 """
    factor = []
    for i, (g1, g2) in enumerate(zip(factor1, factor2)):
        factor.append(g1 & g2)
    return factor


def bin_list_remove(factor1, factor2):
    """ 求补集 """
    factor = factor1.copy()
    for i, (g1, g2) in enumerate(zip(factor1, factor2)):
        if g1 == g2 and g1 == 1:
            factor[i] = 0
    return factor


"""
快速排序算法
"""


class LabelList(object):
    """ 快速排序算法个体类，为拓展个体信息，其中value为排序主键 """

    def __init__(self):
        self.value = []
        self.label = []

    def already(self):
        return len(self.value) == len(self.label)


def partition(label_list=LabelList(), i=0, j=0):
    """ 分区函数 """
    p = label_list.value[i]
    m = i
    for k in range(i + 1, j + 1):
        if label_list.value[k] < p:
            m += 1
            label_list.value[k], label_list.value[m] = label_list.value[m], label_list.value[k]
            label_list.label[k], label_list.label[m] = label_list.label[m], label_list.label[k]
    label_list.value[i], label_list.value[m] = label_list.value[m], label_list.value[i]
    label_list.label[i], label_list.label[m] = label_list.label[m], label_list.label[i]
    return m


def quick_sort(label_list=LabelList(), low=0, high=0):
    """ 
    快速排序算法主体 
    使用示例
    label_list = LabelList()
    label_list.value = [1,4,2,5]
    label_list.label = ['1','4','2','5']
    quickSort(label_list,0,len(label_list)-1)
    """
    if low < high:
        m = partition(label_list, low, high)
        quick_sort(label_list, low, m - 1)
        quick_sort(label_list, m + 1, high)


"""
遗传算法的主体部分
"""


def load_data(data_path=os.path.join("result"), filename='result.csv'):
    """ 导入数据 """
    csv_path = os.path.join(data_path, filename)
    if not os.path.exists(csv_path):
        print("Path does not exist!")
        return DataFrame()
    else:
        return pd.read_csv(csv_path, index_col=0)


def save_data(dataframe, data_path=os.path.join("result"), filename='result.csv'):
    """ 导出数据 """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    csv_path = os.path.join(data_path, filename)
    return dataframe.to_csv(csv_path)


def unsigned(num):
    """ 取绝对值 """
    if num < 0:
        return 0
    else:
        return num


class PredictRecorder(object):
    """docstring for PredictRecorder"""

    def __init__(self, accuracy_fun=None, filename='result.csv',
                 data_path=os.path.join("result")):
        """ 基本参数初始化 """
        self.fitness_list = {}
        self.duo_str = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                        "u", "v"]
        self.duo2binStr = {"0": "00000", "1": "00001", "2": "00010", "3": "00011", "4": "00100", "5": "00101",
                           "6": "00110", "7": "00111", "8": "01000", "9": "01001", "a": "01010", "b": "01011",
                           "c": "01100", "d": "01101", "e": "01110", "f": "01111", "g": "10000", "h": "10001",
                           "i": "10010", "j": "10011", "k": "10100", "l": "10101", "m": "10110", "n": "10111",
                           "o": "11000", "p": "11001", "q": "11010", "r": "11011", "s": "11100", "t": "11101",
                           "u": "11110", "v": "11111"}

        self.filename = filename

        self.last_best_factor = None
        self.last_best_fitness = None
        self.last_best_chromosome_str = None

        self.accuracy_fun = accuracy_fun
        self.fitness_sort = []
        self.chromosomes_str_sort = []
        self.count_sort = []
        self.solution_sort = []
        self.accuracy_sort = []

        self.LOAD_DATA_PATH = data_path
        self.SAVE_DATA_PATH = data_path

    def __bin2duoList(self, binary, duotricemary):
        """ Duotricemary Notation 三十二进制计数法 修改列表 duotricemary """
        if duotricemary is None:
            duotricemary = []
        if binary is None:
            binary = []
        if len(binary) > 5:
            self.__bin2duoList(binary[0:len(binary) - 5], duotricemary)
        bin_temp = binary[unsigned(len(binary) - 5):len(binary)]
        count = sum([j * (2 ** (len(bin_temp) - 1 - i)) for (i, j) in enumerate(bin_temp)])
        duotricemary.append(self.duo_str[count])

    def binList2duoList(self, binary=None):
        duo = []
        self.__bin2duoList(binary, duo)
        return duo

    def binList2duoStr(self, bin_list=None):
        """ Duotricemary Notation 三十二进制计数法 返回字符串 """
        if bin_list is None:
            bin_list = [1]
        duo = self.binList2duoList(bin_list)
        for (i, j) in enumerate(duo):
            if j != "0":
                break
            duo[i] = ""
        if duo[len(duo) - 1] == "":
            duo[len(duo) - 1] = "0"
        duotricemary = map(str, duo)
        return ''.join(duotricemary)

    def duoStr2binList(self, duo_str="", n=None):
        """ 三十二进制字符串转二进制列表 返回列表binary """
        if type(duo_str) != str:
            duo_str = str(duo_str)
        binary = []
        for i in duo_str:
            if i != ' ':
                for j in self.duo2binStr[i]:
                    binary.append(int(j))

        if n is not None:
            if len(binary) < n:
                binary = [0 for _ in range(n - len(binary))] + binary
            else:
                binary = binary[-n:]
        return binary

    def append(self, chromosome, result_list):
        """ 附加数据 """
        if type(chromosome) == list:
            self.fitness_list[self.binList2duoStr(chromosome)] = result_list
        if type(chromosome) == str:
            self.fitness_list[chromosome] = result_list

    def is_store(self, chromosome):
        """ 查询列表是否存在 """
        if type(chromosome) == list:
            return self.binList2duoStr(chromosome) in self.fitness_list
        if type(chromosome) == str:
            return chromosome in self.fitness_list
        else:
            return str(chromosome) in self.fitness_list

    def store_self(self, isprint=False, filename=None):
        """ 存储历史记录 """
        if isprint:
            print("Saving Recorded Data!")
        data = DataFrame()
        column = ["chromosome", "fitness", "solution", "count", "accuracy"]
        datadict = {i: [] for i in column}
        datadict["chromosome"] = [j for j in self.fitness_list]
        datadict["fitness"] = [self.fitness_list[j][0] for j in self.fitness_list]
        datadict["solution"] = [self.fitness_list[j][1] for j in self.fitness_list]
        datadict["count"] = [self.fitness_list[j][2] for j in self.fitness_list]
        datadict["accuracy"] = [self.fitness_list[j][3] for j in self.fitness_list]
        for i in column:
            data[i] = datadict[i]

        if filename is None:
            save_data(data, filename=self.filename)
        else:
            save_data(data, filename=filename)

    def store_self_by_chromosome_sort(self):
        """ 存储单特征记录 """
        data = DataFrame()
        column = ["chromosome", "fitness", "solution", "count", "accuracy"]
        datadict = {i: [] for i in column}
        datadict["chromosome"] = self.chromosomes_str_sort
        datadict["fitness"] = self.fitness_sort
        datadict["solution"] = self.solution_sort
        datadict["count"] = self.count_sort
        datadict["accuracy"] = self.accuracy_sort
        for i in column:
            data[i] = datadict[i]
        save_data(data, data_path=self.SAVE_DATA_PATH, filename=self.filename)

    def restore(self, trace_best_accuracy=None, is_count_one=False, is_show=True):
        """ 恢复历史记录 """
        if is_show:
            print("Restore recorded history!")
        data = load_data(data_path=self.LOAD_DATA_PATH, filename=self.filename)
        if len(data) != 0:
            self.fitness_list.clear()
            label_list = LabelList()
            for row in data.itertuples(index=True, name='Pandas'):
                label_list.value.append(getattr(row, "accuracy"))
                label_list.label.append(
                    [str(getattr(row, "chromosome").replace(" ", "")), getattr(row, "solution"),
                     getattr(row, "count"),
                     self.accuracy_fun(getattr(row, "accuracy")) if callable(self.accuracy_fun)
                     else getattr(row, "accuracy")])
                self.fitness_list[str(getattr(row, "chromosome")).replace(" ", "")] = [
                    self.accuracy_fun(getattr(row, "accuracy")) if callable(self.accuracy_fun)
                    else getattr(row, "accuracy"),
                    getattr(row, "solution"), getattr(row, "count"), getattr(row, "accuracy")]

            quick_sort(label_list, 0, len(label_list.value) - 1)

            self.fitness_sort.clear()
            self.chromosomes_str_sort.clear()
            self.solution_sort.clear()
            self.count_sort.clear()
            self.accuracy_sort.clear()
            if is_count_one:
                for i, j in enumerate(label_list.label):
                    if int(j[2]) == 1:
                        self.chromosomes_str_sort.append(j[0])
                        self.solution_sort.append(j[1])
                        self.count_sort.append(j[2])
                        self.fitness_sort.append(j[3])
                        self.accuracy_sort.append(label_list.value[i])
            else:
                for i, j in enumerate(label_list.label):
                    self.chromosomes_str_sort.append(j[0])
                    self.solution_sort.append(j[1])
                    self.count_sort.append(j[2])
                    self.fitness_sort.append(j[3])
                    self.accuracy_sort.append(label_list.value[i])
            start_index = 0
            if trace_best_accuracy is not None:
                print("Try to find best accuracy \"", trace_best_accuracy, "\" in recorder dataset!")
                while abs(label_list.value[len(label_list.value) - 1 - start_index] - trace_best_accuracy) > 5e-10:
                    start_index = start_index + 1

            if self.accuracy_fun is None:
                self.last_best_fitness = label_list.label[len(label_list.label) - 1 - start_index][3]
            else:
                self.last_best_fitness = self.accuracy_fun(label_list.value[len(label_list.value) - 1 - start_index])
            self.last_best_chromosome_str = label_list.label[len(label_list.label) - 1 - start_index][0]
            self.last_best_factor = self.duoStr2binList(self.last_best_chromosome_str)
            if is_show:
                self.show_self()
        else:
            print("There is not data recorded !")

    def show_self(self):
        """ 打印历史记录 """
        print("********************************************** History ************************************************")
        print("Chromosome Fitness Solution Count Accuracy")
        for j in self.fitness_list:
            print(j, "\t", self.fitness_list[j][0], "\t", self.fitness_list[j][1], "\t", self.fitness_list[j][2], "\t",
                  self.fitness_list[j][3])
        print("Last best fitness_list:", self.last_best_fitness)
        print("Last best chromosome:", self.last_best_chromosome_str)
        print("Last best factor count:", sum(self.last_best_factor))  # ," <未补位二进制列表>"


def plot_interactive_mode_base():
    """ 开启交互模式，不再阻塞程序 """
    plt.ion()
    plt.figure(1)


class TraceRecorder(object):
    """Record trace"""

    def __init__(self, filename='trace.csv', data_path=os.path.join("result")):
        self.iter = list()
        self.best_fitness = list()
        self.best_solution = list()
        self.best_accuracy = list()

        self.current_best_fitness = list()
        self.current_best_accuracy = list()
        self.current_best_solution = list()
        self.avg_fitness = list()
        # self.best_chromosomes = list()
        # self.best_chromosome_str_list = list()
        self.last_best_fitness = None
        self.last_best_accuracy = None
        self.filename = filename
        self.LOAD_DATA_PATH = data_path
        self.SAVE_DATA_PATH = data_path

    def append(self, i, best_fitness, best_accuracy, best_solution,
               current_best_fitness, current_best_accuracy, current_best_solution,
               avg_fitness, best_chromosome):
        """ 附加数据 """
        self.iter.append(i)
        self.best_fitness.append(best_fitness)
        self.best_solution.append(best_solution)
        self.best_accuracy.append(best_accuracy)

        self.current_best_fitness.append(current_best_fitness)
        self.current_best_solution.append(current_best_solution)
        self.current_best_accuracy.append(current_best_accuracy)
        self.avg_fitness.append(avg_fitness)
        # self.best_chromosomes.append(best_chromosome.copy())
        # binary = map(str, best_chromosome)
        # binary_str = ''.join(binary)
        # self.best_chromosome_str_list.append(binary_str)

    def store_self(self):
        """ 存储数据 """
        data = DataFrame()
        data['best_fitness'] = self.best_fitness
        data['best_accuracy'] = self.best_accuracy
        data['current_best_fitness'] = self.current_best_fitness
        data['current_best_accuracy'] = self.current_best_accuracy
        save_data(data, data_path=self.SAVE_DATA_PATH, filename=self.filename)

    def restore(self):
        """ 导入数据 """
        print("Restore recorded data of trace!")
        data = load_data(data_path=self.LOAD_DATA_PATH, filename=self.filename)
        if len(data) != 0:
            self.best_fitness = list(data['best_fitness'])
            self.current_best_fitness = list(data['current_best_fitness'])
            self.best_accuracy = list(data['best_accuracy'])
            self.current_best_accuracy = list(data['current_best_accuracy'])
            self.last_best_fitness = self.best_fitness[len(self.best_fitness) - 1]
            self.last_best_accuracy = self.best_accuracy[len(self.best_accuracy) - 1]
            print("Best fitness_list of trace:", self.last_best_fitness,
                  "\nBest accuracy of trace:", self.last_best_accuracy)
        else:
            print("There is not trace data recorded !")

    def show_trace(self, is_accuracy=True):
        """ 显示进化曲线 """
        plt.cla()
        if is_accuracy:
            plt.plot(self.best_accuracy)
            plt.plot(self.current_best_accuracy)
        else:
            plt.plot(self.best_fitness)
            plt.plot(self.current_best_fitness)
        plt.draw()
        plt.pause(0.001)  # 图片展示后就会关闭，适当延迟以看效果

    def show_search_progress(self, fun=None):
        """ 显示在适应度函数上的搜索过程 """
        if callable(fun):
            plt.cla()
            x = [i for i in np.arange(0, 10.01, 0.01)]
            y = [fun(xi)[1] for xi in x]
            plt.plot(x, y)
            plt.plot(self.current_best_solution,
                     self.current_best_fitness, "rx")
            plt.draw()
            plt.pause(0.001)  # 图片展示后就会关闭，适当延迟以看效果

    def print_info(self):
        """ 打印信息 """
        print("Information of optimal solution :",
              "\n\tbest fitness :", self.best_fitness[len(self.best_fitness) - 1],
              "\n\tbest accuracy :", self.best_accuracy[len(self.best_accuracy) - 1],
              "\n\tcurrent best fitness :", self.current_best_fitness[len(self.current_best_fitness) - 1],
              "\n\tcurrent best accuracy :", self.current_best_accuracy[len(self.current_best_accuracy) - 1])


class Population(object):
    """Population Class"""

    def __init__(self, size_pop=8, history_percent=1 / 50, len_chromosome=16, max_length=16, min_length=1,
                 best_factor_temp=None, parallel_num=1, function=None, chromosome_checker=None,
                 recorder_file="result.csv", current_chromosome_file='current_chromosome.csv',
                 data_path=os.path.join("result"), accuracy_fun=None):
        """ 基本的参数设置 """
        if best_factor_temp is None:
            best_factor_temp = []

        self.chromosome_list = list()
        self.fitness_list = list()
        self.solution_list = list()
        self.chromosome_str_list = list()
        self.already = 0
        self.size_pop = size_pop
        self.history_percent = history_percent
        self.history_length = int(self.size_pop * history_percent)
        self.len_chromosome = len_chromosome
        self.parallel_num = parallel_num
        self.function = function
        self.chromosome_checker = chromosome_checker
        self.max_length = max_length
        self.min_length = min_length
        if best_factor_temp is None:
            self.best_factor_temp = []
        else:
            self.best_factor_temp = best_factor_temp.copy()
        self.accuracy_list = list()
        self.count_list = list()
        for i in range(size_pop):
            self.solution_list.append(0)
            self.fitness_list.append(0)
            self.count_list.append(0)
            self.accuracy_list.append(0)
            self.chromosome_str_list.append("")
        self.is_init = (len(self.best_factor_temp) != 0)  # (self.len_chromosome == len(self.best_factor_temp))
        if self.is_init:
            print("Have init value!")

        self.recorder = PredictRecorder(accuracy_fun=accuracy_fun, filename=recorder_file)
        self.current_recorder = PredictRecorder(accuracy_fun=accuracy_fun, filename="current_" + recorder_file)
        self.current_chromosome_file = current_chromosome_file
        self.LOAD_DATA_PATH = data_path
        self.SAVE_DATA_PATH = data_path

        self.accuracy_fun = accuracy_fun
        self.calc_num = 0

        self.best_fitness = None
        self.best_index = None
        self.best_chromosome = None
        self.best_chromosome_str = None
        self.best_solution = None
        self.best_accuracy = None

        self.best_factor = None
        self.chromosome_list_sort = None
        self.accuracy_sort = None
        self.fitness_sort = None

        self.current_fitness_list = {}

    def clear(self, n):
        self.chromosome_list = list()
        self.fitness_list = list()
        self.solution_list = list()
        self.chromosome_str_list = list()
        self.count_list = list()
        self.accuracy_list = list()
        self.current_recorder.fitness_list = {}
        for i in range(n):
            self.fitness_list.append(0)
            self.solution_list.append(0)
            self.count_list.append(0)
            self.accuracy_list.append(0)
            self.chromosome_str_list.append("")

    def restore(self, trace_best_accuracy=None, is_count_one=False, is_show=True):
        self.recorder.restore(trace_best_accuracy, is_count_one, is_show)
        self.best_chromosome_str = self.recorder.last_best_chromosome_str
        self.best_fitness = self.recorder.last_best_fitness
        if type(self.recorder.last_best_factor) == list():
            self.recorder.last_best_factor = self.recorder.last_best_factor[-self.len_chromosome:]
        self.best_factor = self.recorder.last_best_factor
        if self.recorder.chromosomes_str_sort is not None:
            self.chromosome_list_sort = [self.recorder.duoStr2binList(s, self.len_chromosome)
                                         for s in self.recorder.chromosomes_str_sort]
            self.accuracy_sort = self.recorder.accuracy_sort
            self.fitness_sort = self.recorder.fitness_sort

    def init(self, trace_best_accuracy=None, use_history=False, use_last=False, history_percent=None):
        """ 初始化初代个体 """
        print("Information of population :",
              "\n\tsize of population :", self.size_pop,
              "\n\tlength of chromosome :", self.len_chromosome,
              "\n\tmax length :", self.max_length,
              "\n\tmin length :", self.min_length)

        self.restore(trace_best_accuracy)
        self.clear(self.size_pop)
        print("**********************************Population Initialization********************************************")
        ''' history > trace best accuracy > temp best factor '''
        count = self.size_pop
        used_best = 0
        length = None
        history = None
        history_count = None

        if use_last:
            self.restore_current_chromosome()
            length = len(self.current_fitness_list)
            history = [i for i in self.current_fitness_list]
            history_count = [self.current_fitness_list[i][2] for i in self.current_fitness_list]
            self.history_percent = 1
        else:
            if use_history:
                length = len(self.recorder.chromosomes_str_sort)
                history = self.recorder.chromosomes_str_sort
                history_count = self.recorder.count_sort

        if use_history or use_last:
            if history_percent is not None:
                history_percent = history_percent if history_percent <= 1 else 1
            else:
                history_percent = self.history_percent

            if history_percent is None:
                self.history_length = self.history_length
            else:
                self.history_length = int(self.size_pop * history_percent)

            for i in range((self.history_length if self.history_length <= length else length)):
                if self.max_length >= history_count[length - i - 1] >= self.min_length:
                    self.chromosome_list.append(self.recorder.duoStr2binList(history[length - i - 1],
                                                                             self.len_chromosome))
                    count = count - 1
            used_best = 1
            print("The size of data of history used is:",
                  self.history_length if self.history_length <= length else length, "!")
            print("The size of individuals which still need to be generated is:", count, "!")
        else:
            if self.recorder.last_best_factor is not None:
                self.best_factor_temp = self.recorder.last_best_factor.copy()
                if len(self.best_factor_temp) < self.len_chromosome:
                    self.best_factor_temp = [0 for _ in range(
                        self.len_chromosome - len(self.best_factor_temp))] + self.best_factor_temp
                self.is_init = True
                print("There is an optimal solution!")
            if self.is_init:
                if type(self.best_factor_temp[0]) == list:
                    for i in self.best_factor_temp:
                        self.chromosome_list.append(i.copy())
                        count = count - 1
                        used_best = used_best + 1
                else:
                    self.chromosome_list.append(self.best_factor_temp.copy())
                    count = count - 1
                    used_best = 1

        for i in range(count):
            print("\rProgress of initialization :{}/{}, the number of last optimal solution:{} \t"
                  .format(i + 1, count, used_best), end="")
            chromosome = list()
            if np.random.random() > 2 / count or not self.is_init:
                flag = 0
                while flag == 0:
                    for j in range(self.len_chromosome):
                        value = 1 if np.random.random() <= ((self.min_length + (
                                self.max_length - self.min_length) * np.random.random()) / self.len_chromosome if int(
                            self.max_length / self.len_chromosome) != 1 else 0.5) else 0
                        chromosome.append(value)
                    if (sum(chromosome) <= self.max_length) and (sum(chromosome) >= self.min_length):
                        if self.chromosome_checker(chromosome) if callable(self.chromosome_checker) else True:
                            flag = 1
                    else:
                        chromosome.clear()
            else:
                # Used the last best value!
                used_best = used_best + 1
                chromosome = self.best_factor_temp.copy()
            self.chromosome_list.append(chromosome)
        self.get_fitness()

    def alone_feature_general(self, n=1):
        self.clear(self.len_chromosome)
        self.restore(is_count_one=True, is_show=False)
        self.clear(n)
        for i in range(n):
            init_list = [0 for _ in range(n)]
            init_list[i] = 1
            self.chromosome_list.append(init_list)
        self.get_fitness(is_sort=True, is_show=True)
        self.recorder.store_self()

    def auxiliary_fitness(self, chromosome):
        """ 适应度附加信息函数获取 """
        fitness_temp = self.fitness_fun(chromosome)
        solution_temp = 0
        if type(fitness_temp) is tuple:
            solution_temp = fitness_temp[0]
            fitness_temp = fitness_temp[1]
        accuracy_temp = fitness_temp
        if callable(self.accuracy_fun):
            accuracy_temp = self.accuracy_fun(accuracy_temp)
        return [accuracy_temp, solution_temp, sum(chromosome), fitness_temp]

    def restore_current_chromosome(self):
        """ 恢复历史记录 """
        print("Restore Last Recorded Data!")
        data = load_data(filename='current_chromosome.csv')
        if len(data) != 0:
            self.current_fitness_list.clear()
            for row in data.itertuples(index=True, name='Pandas'):
                self.current_fitness_list[str(getattr(row, "chromosome")).replace(" ", "")] = \
                    [getattr(row, "fitness"), getattr(row, "solution").replace(" ", ""),
                     getattr(row, "count"), getattr(row, "accuracy")]
        else:
            print("There is not last data recorded !")

    def store_current_chromosome(self, isprint=False):
        """ 存储历史记录 """
        if isprint:
            print("Saving Recorded Current Chromosome!")
        data = DataFrame()
        column = ["chromosome", "fitness", "solution", "count", "accuracy"]
        datadict = {i: [] for i in column}
        datadict["chromosome"] = self.chromosome_str_list
        datadict["fitness"] = self.fitness_list
        datadict["solution"] = self.solution_list
        datadict["count"] = self.count_list
        datadict["accuracy"] = self.accuracy_list
        for i in column:
            data[i] = datadict[i]
        save_data(data, filename="current_chromosome.csv")

    def get_one_fitness(self, j, chromosome):
        """ 获取单个染色体对应的适应度 """
        chromosome_str = self.recorder.binList2duoStr(chromosome)
        if self.recorder.is_store(chromosome_str):
            fitness_temp = self.recorder.fitness_list[chromosome_str]
        else:
            # self.calc_num = self.calc_num + 1
            # calc_num_temp = self.calc_num
            # print("\r calc num : " + str(calc_num_temp), end="")
            fitness_temp = self.auxiliary_fitness(chromosome)
        try:
            self.fitness_list[j] = fitness_temp[0]
            self.solution_list[j] = fitness_temp[1]
            self.count_list[j] = fitness_temp[2]
            self.accuracy_list[j] = fitness_temp[3]
            self.chromosome_str_list[j] = chromosome_str
            self.recorder.append(chromosome_str, fitness_temp)
        except IndexError:
            print("IndexError index:", j)
            print("length of fitness temp:", len(fitness_temp))
            print("length of fitness:", len(self.fitness_list))
            print("length of solution:", len(self.solution_list))
            print("length of chromosome_str:", len(self.chromosome_str_list))
            print("length of count list:", len(self.count_list))
            print("length of accuracy list:", len(self.accuracy_list))
        else:
            pass
        finally:
            pass
        self.already = self.already - 1
        if not self.current_recorder.is_store(chromosome_str):
            self.current_recorder.append(chromosome_str, fitness_temp)
        return fitness_temp[0]

    def get_fitness(self, is_sort=False, is_show=False):
        """ 获取染色体列表对应的适应度列表 """
        for j, chromosome in enumerate(self.chromosome_list):
            if is_show:
                print("\rGet fitness progress:{}/{}".format(j + 1, len(self.chromosome_list)), end="")
            threading.Thread(target=self.get_one_fitness, args=(j, chromosome)).start()
            self.already = self.already + 1
            while self.already == self.parallel_num:
                sleep(0.1)
        while self.already != 0:
            sleep(0.1)
        if is_sort:
            self.sort_chromosome()

        self.best_fitness = min(self.fitness_list)
        self.best_index = self.fitness_list.index(self.best_fitness)
        self.best_chromosome = self.chromosome_list[self.best_index].copy()
        self.best_chromosome_str = self.chromosome_str_list[self.best_index]
        self.best_solution = self.solution_list[self.best_index]
        self.best_accuracy = self.accuracy_list[self.best_index]
        self.store_current_chromosome()
        self.recorder.store_self()
        self.current_recorder.store_self()
        return self.fitness_list

    def store_current_recorder(self, filename):
        self.current_recorder.store_self(filename=filename)

    def sort_chromosome(self):
        label_list = LabelList()
        # 为维持源数据的排列顺序选择使用copy()
        label_list.value = self.fitness_list.copy()
        label_list.label = self.chromosome_list.copy()
        quick_sort(label_list, 0, len(label_list.value) - 1)
        self.fitness_sort = label_list.value
        self.chromosome_list_sort = label_list.label

    def get_init_best_chromosome(self, is_count_one=True):
        self.accuracy_fun = None
        self.clear(self.len_chromosome)
        self.restore(is_count_one=is_count_one, is_show=False)
        best_fitness = 0
        best_chromosome = [0 for _ in range(self.len_chromosome)]
        candidate_chromosomes = [self.chromosome_list_sort[len(self.chromosome_list_sort) - 1 - i] for i in
                                 range(len(self.chromosome_list_sort))]
        candidate_chromosomes_temp = candidate_chromosomes.copy()
        evolve_flag = True
        while evolve_flag and sum(best_chromosome) < self.len_chromosome and best_fitness < 1:
            evolve_flag = False
            for candidate_chromosome in candidate_chromosomes:
                best_chromosome_temp = bin_union_list(best_chromosome,
                                                      self.recorder.duoStr2binList(candidate_chromosome,
                                                                                   self.len_chromosome))
                print("\rBest fitness is: {}, count of best chromosome: {}\t\t".
                      format(best_fitness, sum(best_chromosome)), end="")
                best_fitness_temp = self.get_one_fitness(0, best_chromosome_temp)
                if best_fitness < best_fitness_temp:
                    best_fitness = best_fitness_temp
                    best_chromosome = best_chromosome_temp
                    candidate_chromosomes_temp.remove(candidate_chromosome)
                    evolve_flag = True
                    if best_fitness >= 1:
                        break
            candidate_chromosomes = candidate_chromosomes_temp.copy()
        self.best_fitness = best_fitness
        self.best_chromosome = best_chromosome
        self.recorder.store_self()

    def simplify_best_chromosome(self):
        self.accuracy_fun = None
        self.clear(self.len_chromosome)
        self.restore(is_show=False)
        best_fitness = self.best_fitness
        best_chromosome = self.best_factor
        evolve_flag = True
        while evolve_flag:
            evolve_flag = False
            for i, bit in enumerate(best_chromosome):
                if bit == 1:
                    best_chromosome_temp = best_chromosome.copy()
                    best_chromosome_temp[i] = 0
                    best_fitness_temp = self.get_one_fitness(0, best_chromosome_temp)
                    if best_fitness_temp > best_fitness:
                        evolve_flag = True
                        best_fitness = best_fitness_temp
                        best_chromosome[i] = 0
        self.best_fitness = best_fitness
        self.best_chromosome = best_chromosome
        self.recorder.store_self()

    def fitness_fun(self, chromosome):
        """ 适应度函数 """
        ret = sum(chromosome) if not callable(self.function) else self.function(chromosome)
        return ret

    def copy(self, old_population):
        """ 同类对象的深拷贝函数 """
        self.chromosome_list = old_population.chromosome_list.copy()
        self.best_index = old_population.best_index
        self.best_fitness = old_population.best_fitness
        self.best_solution = old_population.best_solution
        self.best_chromosome_str = old_population.best_chromosome_str
        self.best_chromosome = old_population.best_chromosome.copy()
        self.fitness_list = old_population.fitness_list.copy()
        self.solution_list = old_population.solution_list.copy()
        self.chromosome_str_list = old_population.chromosome_str_list.copy()
        self.size_pop = old_population.size_pop
        self.len_chromosome = old_population.len_chromosome
        self.function = old_population.function
        self.best_factor_temp = old_population.best_factor_temp
        self.accuracy_list = old_population.accuracy_list.copy()
        self.count_list = old_population.count_list.copy()
        self.recorder = old_population.recorder
        self.parallel_num = old_population.parallel_num
        self.min_length = old_population.min_length
        self.max_length = old_population.max_length
        self.calc_num = old_population.calc_num
        self.accuracy_fun = old_population.accuracy_fun

    def get_best_chromosome(self, trace_best_accuracy=None):
        self.recorder.restore(trace_best_accuracy)
        return self.recorder.last_best_factor


def select_copy(value, index):
    """ 部分复制 """
    return [value[i] for i in index]


class CrossType(Enum):
    SINGLE_POINT = 1
    UNIFORMITY = 2


class GAToolBox(object):
    """GAToolBox Main Body"""

    def __init__(self, size_pop=10, max_gen=200, len_chromosome=16, max_length=None,
                 min_length=None, p_cross=0.6, p_exchange=0.3, p_mutation=0.3, gap=0.9,
                 min_value=0, factor_best_temp=None, parallel_num=1, function=None,
                 accuracy_fun=None, chromosome_checker=None, trace_file="trace.csv", recorder_file="result.csv",
                 current_chromosome_file='current_chromosome.csv', data_path=os.path.join("result")):
        """ 基本的参数设置 """
        self.min_value = min_value
        self.size_pop = size_pop
        self.max_gen = max_gen
        self.p_cross = p_cross
        self.len_chromosome = len_chromosome
        self.parallel_num = parallel_num
        self.p_exchange = p_exchange
        self.p_mutation = p_mutation
        self.gap = gap
        self.function = function
        self.max_length = max_length if max_length is not None else len_chromosome
        self.min_length = min_length if min_length is not None else 0

        self.best_accuracy = 0
        self.avg_fitness = 0
        self.best_fitness = 0
        self.best_chromosome = []
        self.best_solution = 0

        if factor_best_temp is None:
            self.factor_best_temp = []
        else:
            self.factor_best_temp = factor_best_temp.copy()

        self.have_run = False
        self.trace = TraceRecorder(filename=trace_file, data_path=data_path)
        self.population = Population(size_pop=self.size_pop, len_chromosome=self.len_chromosome,
                                     max_length=self.max_length, min_length=self.min_length,
                                     best_factor_temp=self.factor_best_temp, parallel_num=self.parallel_num,
                                     function=self.function, accuracy_fun=accuracy_fun,
                                     chromosome_checker=chromosome_checker, recorder_file=recorder_file,
                                     current_chromosome_file=current_chromosome_file, data_path=data_path)
        self.chromosome_checker = chromosome_checker

    def select(self, population, size_pop):
        """ 采用轮盘赌选择 """
        wheel_fitness_base = [i if i != 0 else 1e-32 for i in population.fitness_list]
        wheel_fitness = list((1 / (np.array(wheel_fitness_base))) / sum(1 / (np.array(wheel_fitness_base))))
        index = list()
        # 旋转size_pop次轮盘选择
        for i in range(0, size_pop - 1, 1):
            print("\rselection progress : {}% \t\t\t".format(int((i + 1) / (size_pop - 1) * 100)), end="")
            pick = np.random.random()
            while pick == 0:
                pick = np.random.random()
            for j in range(0, size_pop, 1):
                pick = pick - wheel_fitness[j]
                if pick < 0:
                    index.append(j)
                    break
        population.chromosome_list = select_copy(population.chromosome_list, index)
        population.fitness_list = select_copy(population.fitness_list, index)
        population.chromosome_list.append(self.best_chromosome.copy())
        population.fitness_list.append(self.best_fitness)
        return population

    def cross(self, p_cross, p_exchange, len_chromosome, chromosomes, size_pop, cross_type=CrossType.UNIFORMITY):
        """ 本程序采用均匀交叉 """
        for i in range(0, int(size_pop * self.gap), 1):
            print("\rcross progress : {}% \t\t\t".format(int((i + 1) / size_pop / self.gap * 100)), end="")
            # 随机选择两个染色体进行交叉
            pick = np.random.random(2)
            while np.fabs(pick[0]-pick[1]) < 1e-4:
                pick = np.random.random(2)
            index = [int(pick[0] * size_pop), int(pick[1] * size_pop)]
            # 根据交叉概率决定是否交叉
            pick = np.random.random()
            while pick == 0:
                pick = np.random.random()
            if pick > p_cross:
                continue
            flag = False
            while not flag:
                # 随机选择两个染色体的交叉位置
                pick = np.random.random()
                while pick == 0:
                    pick = np.random.random()
                chromosome_1 = chromosomes[index[0]].copy()
                chromosome_2 = chromosomes[index[1]].copy()
                if cross_type is CrossType.UNIFORMITY:
                    # 均匀交叉
                    for j, (g1, g2) in enumerate(zip(chromosome_1, chromosome_2)):
                        do_exchange = True if np.random.random() < p_exchange else False
                        if do_exchange:
                            chromosome_1[j], chromosome_2[j] = g2, g1
                elif cross_type is CrossType.SINGLE_POINT:
                    # 单点交叉
                    pos = int(pick * len_chromosome)
                    for j, (g1, g2) in enumerate(zip(chromosome_1, chromosome_2)):
                        if i <= pos:
                            chromosome_1[j], chromosome_2[j] = g2, g1
                        else:
                            break
                if (sum(chromosome_1) <= self.max_length) and (sum(chromosome_2) <= self.max_length) and (
                        sum(chromosome_1) >= self.min_length) and (sum(chromosome_2) >= self.min_length):
                    if (self.chromosome_checker(chromosome_1) and self.chromosome_checker(chromosome_2)) \
                            if callable(self.chromosome_checker) else True:
                        chromosomes[index[0]] = chromosome_1.copy()
                        chromosomes[index[1]] = chromosome_2.copy()
                        flag = True
        return chromosomes

    def mutation(self, p_mutation, len_chromosome, chromosomes, size_pop):
        """ 本程序采用均匀变异 """
        for i in range(0, int(size_pop * self.gap), 1):
            print("\rmutation progress : {}% \t\t\t".format(int((i + 1) / size_pop / self.gap * 100)), end="")
            # 随机选择一个染色体进行变异
            pick = np.random.random()
            index = int(pick * size_pop)
            # 决定是否变异
            pick = np.random.random()
            while pick == 0:
                pick = np.random.random()
            if pick > p_mutation:
                continue
            flag = False
            while not flag:
                chromosome_temp = chromosomes[index].copy()
                mutation_count = sum(chromosome_temp)
                for j, bit in enumerate(chromosome_temp):
                    no_flip = True if np.random.random() > p_mutation else False
                    if no_flip:
                        continue
                    # 为确保染色体中基因为1的最多位数或者特征个数
                    if mutation_count < len_chromosome / 2:
                        if np.random.random() <= (1 if bit == 1
                                                  else mutation_count / (len_chromosome - mutation_count)):
                            chromosome_temp[j] = bit ^ 1
                    else:
                        if np.random.random() <= (1 if bit == 0
                                                  else (len_chromosome - mutation_count) / mutation_count):
                            chromosome_temp[j] = bit ^ 1
                if (sum(chromosome_temp) <= self.max_length) and (sum(chromosome_temp) >= self.min_length):
                    if self.chromosome_checker(chromosome_temp) if callable(self.chromosome_checker) else True:
                        chromosomes[index] = chromosome_temp.copy()
                        flag = True
        return chromosomes

    def run(self, iteration=None, is_show_trace=False, if_show_search=False, use_history=False,
            use_last=False, history_percent=None):
        """ 遗传算法实现主程序 """
        print("******************************************* Genetic Algorithm *****************************************")
        start_time = tm.time()
        if use_last:
            self.trace.restore()
        if not self.have_run:
            self.population.init(self.trace.last_best_accuracy, use_history=use_history, use_last=use_last,
                                 history_percent=history_percent)
            self.best_fitness = self.population.best_fitness
            # 由于种群的最优值 <= 当前已有的最优值 所以需要使用copy()保存当前已有的最优解
            self.best_chromosome = self.population.best_chromosome.copy()
            self.best_accuracy = self.population.best_accuracy
            self.best_solution = self.population.best_solution
            self.avg_fitness = sum(self.population.fitness_list) / self.population.size_pop
            self.trace.append(0, self.best_fitness, self.best_accuracy, self.best_solution,
                              self.population.best_fitness, self.population.best_accuracy,
                              self.population.best_solution, self.avg_fitness, self.best_chromosome)
        self.trace.store_self()
        if is_show_trace:
            if not self.have_run:
                # 初始化曲线显示
                plot_interactive_mode_base()

        if iteration is None:
            iteration = self.max_gen
        print("\n******************************************* Solving **************************************************"
              "*")
        for i in range(iteration):
            self.population = self.select(self.population, self.size_pop)
            self.avg_fitness = sum(self.population.fitness_list) / self.population.size_pop
            self.population.chromosome_list = self.cross(self.p_cross, self.p_exchange, self.len_chromosome,
                                                         self.population.chromosome_list, self.size_pop)
            self.population.chromosome_list = self.mutation(self.p_mutation, self.len_chromosome,
                                                            self.population.chromosome_list, self.size_pop)

            self.population.get_fitness()
            if self.population.best_fitness < self.best_fitness:
                # 由于种群的最优值 <= 当前已有的最优值 所以需要使用copy()保存当前已有的最优解
                self.best_chromosome = self.population.best_chromosome.copy()
                self.best_solution = self.population.best_solution
                self.best_fitness = self.population.best_fitness
                self.best_accuracy = self.population.best_accuracy
            self.avg_fitness = sum(self.population.fitness_list) / self.population.size_pop

            print("\rCurrent best fitness_list: {}, current best accuracy: {}, progress of evolution: {}/{}"
                  .format(self.best_fitness,
                          self.best_accuracy, i + 1,
                          self.max_gen), end="")

            self.trace.append(i, self.best_fitness, self.best_accuracy, self.best_solution,
                              self.population.best_fitness, self.population.best_accuracy,
                              self.population.best_solution, self.avg_fitness, self.best_chromosome)
            self.trace.store_self()
            if is_show_trace:
                if if_show_search:
                    # 显示在适应度函数上的搜索过程
                    self.trace.show_search_progress(self.function)
                else:
                    # 显示进化曲线
                    self.trace.show_trace()
            if self.best_fitness <= self.min_value:
                break

        self.max_gen = self.max_gen - iteration
        end_time = tm.time()
        print("\n", end="")
        self.trace.print_info()
        self.have_run = True
        print("Cost time:", end_time - start_time, "s")
        return True

    def get_best_chromosome(self):
        """ 获取最佳特征字符串 """
        self.trace.restore()
        return self.population.get_best_chromosome(self.trace.last_best_accuracy)
