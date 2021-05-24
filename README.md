## GAToolBox

在计算机科学和运筹学中，遗传算法（Genetic Algorithm，GA）是受自然选择过程启发的一种元启发法，它属于进化算法（Evolutionary Algorithms, EA）的一种。 遗传算法依赖于诸如突变，交叉和选择等受生物学启发的操作，其通常用于为优化和搜索问题提供高质量解决方案。

> In computer science and operations research, a genetic algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover and selection.

首先需要解决染色体的编码问题，常见的染色体编码有2进制、8进制、10进制、16进制。由于其中二进制的位编辑灵活简单，所以只进行了二进制编码的实现。其次是交叉和突变，交叉的可选类型较多，针对其实用性本文实现了单点交叉以及均匀交叉。而突变则是根据突变概率对于每一位进行变异操作。同时为了实现历史计算结果存储、上一代种群计算结果存储以及历史进化曲线存储，以保证程序中断后可以在原有基础上继续搜索进化，避免重复计算，同时实现了32进制与2进制数据间转换以方便染色体字的存储，主要针对计算量较大的情况进行的改进。同时为了特征筛选的需求，对于均匀交叉以及突变操作进行了优化以保证染色体中位为1的个数在允许范围内（也就是特征个数在允许范围内）。示例代码如下：

```python
import numpy as np
from GAToolBox import GAToolBox


def fitness(factor):
    """
    用户的适应度函数
    """
    if type(factor) is list:
        binary = map(str, factor)
        binary_str = ''.join(binary)
        if len(binary_str) > 16:
            print(binary_str, "chromosome_list is too long !")
        dec = float(int(binary_str, 2)) / 65535.0 * 10.0
        return dec, sum([j * np.cos((j + 1) * dec + j) for j in range(1, 6)])
    else:
        return factor, sum([j * np.cos((j + 1) * factor + j) for j in range(1, 6)])


def chromosome_checker(factor):
    """
    用户的染色体检查器
    """
    return True


if __name__ == "__main__":
    GA = GAToolBox(min_value=-13, size_pop=10, max_gen=100, p_cross=0.6, p_exchange=0.3, p_mutation=0.4,
                   function=fitness, chromosome_checker=chromosome_checker)
    GA.run(is_show_trace=True, if_show_search=True)

```

运行后可看出遗传算法已找到最优解，现在进行参数讲解：
- size_pop: 种群大小，默认为10
- max_gen: 进化代数，默认为200
- len_chromosome: 染色体长度，默认为16: 
- max_length: 染色体中位为1个数的最大值，默认与len_chromosome保持一致
- min_length: 染色体中位为1个数的最小值，默认为0: 
- p_cross: 两个染色体相互交叉的概率，默认为0.6
- p_exchange: 使用均匀交叉时每位的交叉概率，默认为0.3
- p_mutation: 每个选中的染色体变异的概率以及每位变异的概率
- gap: 隔代差异，即每次进化保留的上一代个体的概率，默认为0.9
- min_value: 最优解的大小，默认为0，该项目中认为最优解适应度函数值最小
- factor_best_temp: 已知的最优解染色体，默认为None
- parallel_num: 使用多线程计算时线程个数
- function: 适应度函数指针，默认为None
- accuracy_fun: 准确率辅助函数，默认为None
- chromosome_checker: 染色体检查器，检查染色体是否可用，默认为None
- trace_file: 进化曲线存储文件名，默认为: "trace.csv"
- recorder_file: 历史计算数据存储文件名，默认为: "result.csv"
- current_chromosome_file: 最新已计算适应度的染色体存储文件名，默认为: "current_chromosome.csv"
- data_path: 全部数据的存储路径，默认为: "./result"

参考论文如下：
> S. -J. Wei, B. Zhang, X. -W. Tan, X. -G. Zhao and D. Ye, "A Real-time Human Activity Recognition Approach with Generalization Performance," 2020 39th Chinese Control Conference (CCC), 2020, pp. 6334-6339, doi: 10.23919/CCC50068.2020.9188860.
