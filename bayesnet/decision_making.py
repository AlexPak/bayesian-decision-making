from typing import List, Dict, Tuple, Any
import numpy as np
import hashlib
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast
from scipy.special import gammaln

#Класс Variable (Обновлённый) для Байесовской сети
#Содержит атрибуты:
      #  name (str): Название переменной (например, "x", "y", "z").
      #  r (int): Количество возможных значений переменной.
      #  values (List[int]): Список возможных значений переменной, начиная с 1.

class Variable:
    
    # Пример __init__:
    #        >>> X = Variable("x", 3)
    #        >>> print(X.name)  # "x"
    #        >>> print(X.r)  # 3
    #        >>> print(X.values)  # [1, 2, 3]
    
    def __init__(self, name: str, r: int):
        self.name: str = name
        self.r: int = r
        self.values = list(range(1, r + 1)) # Cписок значений от 1 до r

    def __str__(self):
        return f"Variable(name={self.name}, num_values={self.r})"

    # Пример __repr__:
    #        >>> X = Variable("x", 3)
    #        >>> X  # Выведет "Variable(name=x, num_values=3)"
 
    def __repr__(self):
        return self.__str__()

    # Пример __hash__:
    #       >>> X = Variable("x", 3)
    #       >>> Y = Variable("y", 2)
    #       >>> variables = {X, Y}
    #       >>> print(variables)  # {Variable(name=x, num_values=3), Variable(name=y, num_values=2)}

    def __hash__(self):
        return hash(self.name)

#Класс Factor вероятностей в Байесовской сети
# Фактор определяет совместное распределение вероятностей для множества случайных переменных.
# Содержит Атрибуты:
    #    vars (List[Variable]): Список переменных, связанных с данным фактором.
    #    table (Dict[Tuple[Any, ...], float]): Таблица вероятностей для каждого набора значений переменных.

# Методы: marginalize(), condition(), multiply() и normalize()

class Factor:
    # vars (List[Variable]): Список переменных, относящихся к этому фактору.
    # table (Dict[Tuple[Any, ...], float]): Таблица вероятностей (0 < P < 1) для комбинаций значений переменных.
    
    def __init__(self, vars: List[Variable], table: Dict[Tuple[Any, ...], float]): # 0 < P < 1 || (E (Pn) = 1)
        # Условие нормализации: Σ P(x) = 1 для дискретных распределений.

        # Пример __init__:
        #    >>> X = Variable("x", 2)
        #    >>> Y = Variable("y", 2)
        #    >>> f = Factor([X, Y], {(1, 1): 0.5, (1, 2): 0.2, (2, 1): 0.2, (2, 2): 0.1})
        #    >>> print(f)
        
        self.vars = vars # Список переменных, включенных в фактор
        self.table = table # Таблица вероятностей

    def __str__(self):

        # Пример __str__:
        #    >>> f = Factor([X, Y], {(1, 1): 0.5, (1, 2): 0.2, (2, 1): 0.2, (2, 2): 0.1})
        #    >>> print(f)
        #    Factor(vars=['x', 'y'], table={(1, 1): 0.5, (1, 2): 0.2, (2, 1): 0.2, (2, 2): 0.1})
        
        return f"Factor(vars={[var.name for var in self.vars]}, table={self.table})"

    def __repr__(self):

        # Возвращает представление объекта, полезное для отладки.
        
        return self.__str__()

    def __hash__(self):

        # Позволяет использовать объекты `Factor` в качестве ключей в словаре или элементах множества.

        # Пример:
        #    >>> factors = {Factor([X], {(1,): 0.6, (2,): 0.4})}
        
        return hash(tuple(self.vars))

    # Метод (3.2)
    def marginalize(self, name: str) -> 'Factor':

        # Выполняет маргинализацию (суммирование) по заданной переменной.

        # Маргинализация удаляет переменную из фактора, суммируя вероятности по всем ее значениям.

        # Аргументы:
        #    name (str): Имя переменной, которую необходимо устранить.

        # Возвращает:
        #    Factor: Новый фактор без указанной переменной.

        # Пример marginalize:
        #    >>> f = Factor([X, Y], {(1, 1): 0.5, (1, 2): 0.2, (2, 1): 0.2, (2, 2): 0.1})
        #    >>> f.marginalize("y")  
        
        new_vars = [var for var in self.vars if var.name != name]
        new_table = {}
        for assignment, prob in self.table.items():
            new_assignment = tuple(value for var, value in zip(self.vars, assignment) if var.name != name)
            new_table[new_assignment] = new_table.get(new_assignment, 0) + prob
        return Factor(new_vars, new_table)

    # Метод (3.3)
    def condition(self, evidence: Dict[str, Any]) -> 'Factor':

        # Условное распределение: фиксирует значения переменных из evidence и удаляет их из фактора.

        # Аргументы:
        #    evidence (Dict[str, Any]): Заданные значения переменных.

        # Возвращает:
        #    Factor: Новый фактор с исключенными переменными.

        # Пример condition:
        #    >>> f = Factor([X, Y], {(1, 1): 0.5, (1, 2): 0.2, (2, 1): 0.2, (2, 2): 0.1})
        #    >>> f.condition({'x': 1})
        #    Factor(vars=['y'], table={(1,): 0.5, (2,): 0.2})
        
        new_table = {}
        for assignment, prob in self.table.items():
            matches = all(assignment[i] == evidence[var.name] for i, var in enumerate(self.vars) if var.name in evidence)
            if matches:
                new_assignment = tuple(assignment[i] for i, var in enumerate(self.vars) if var.name not in evidence)
                new_table[new_assignment] = prob
        new_vars = [var for var in self.vars if var.name not in evidence]
        return Factor(new_vars, new_table)

    def multiply(self, other: 'Factor') -> 'Factor':

        # Перемножает два фактора по общим переменным.

        # Умножение объединяет вероятностные таблицы двух факторов, перемножая совпадающие элементы.

        # Аргументы:
        #    other (Factor): Другой фактор для умножения.

        # Возвращает:
        #    Factor: Новый фактор с объединенной таблицей.

        # Пример multiply:
        #    >>> f1 = Factor([X], {(1,): 0.6, (2,): 0.4})
        #    >>> f2 = Factor([Y, X], {(1, 1): 0.5, (1, 2): 0.8, (2, 1): 0.5, (2, 2): 0.2})
        #    >>> f1.multiply(f2)
        #    Factor(vars=['y', 'x'], table={(1, 1): 0.3, (1, 2): 0.32, (2, 1): 0.3, (2, 2): 0.08})
        
        new_vars = list({var.name: var for var in self.vars + other.vars}.values())
        new_table = {}
        common_vars = set(var.name for var in self.vars) & set(var.name for var in other.vars)
        
        for assignment1, prob1 in self.table.items():
            for assignment2, prob2 in other.table.items():
                if all(assignment1[i] == assignment2[j] for i, var1 in enumerate(self.vars) for j, var2 in enumerate(other.vars) if var1.name == var2.name):
                    new_assignment = tuple(assignment1[i] if var.name in common_vars else assignment2[j] for i, var in enumerate(self.vars) for j, other_var in enumerate(other.vars))
                    new_table[new_assignment] = prob1 * prob2
        return Factor(new_vars, new_table)

    def normalize(self) -> 'Factor':

        # Нормализует таблицу вероятностей так, чтобы их сумма равнялась 1.

        # Возвращает:
        #    Factor: Новый фактор с нормализованными вероятностями.

        # Исключение:
        #    ValueError: Если сумма вероятностей равна 0.

        # Пример:
        #    >>> f = Factor([X], {(1,): 3, (2,): 1})
        #    >>> f.normalize()
        #    Factor(vars=['x'], table={(1,): 0.75, (2,): 0.25})
        
        total_prob = sum(self.table.values())
        if total_prob == 0:
            raise ValueError("Total probability is zero; normalization is impossible.")
        normalized_table = {assignment: prob / total_prob for assignment, prob in self.table.items()}
        return Factor(self.vars, normalized_table)

# Класс BayesianNetwork принятия решений

# Включает в себя вероятностные переменные, факторы, граф связей и функции полезности.

class BayesianNetwork:
    def __init__(self, vars: List[Variable], factors: List[Factor], edges: List[Tuple[str, str]], utilities: Dict[str, Dict[Tuple[Any, ...], float]]):

        # Содержит аргументы __init__:
        #    vars (List[Variable]): Список переменных сети.
        #    factors (List[Factor]): Список факторов (таблиц вероятностей).
        #    edges (List[Tuple[str, str]]): Список рёбер графа, задающих зависимости между переменными.
        #    utilities (Dict[str, Dict[Tuple[Any, ...], float]]): Функции полезности.
            
        self.vars = {var.name: var for var in vars}
        self.factors = factors
        self.graph = nx.DiGraph(edges)
        self.utilities = utilities

    def normalize_factors(self):
        
        # Нормализация факторов. Сумма вероятностей должна быть равна 1.
        
        self.factors = [factor.normalize() for factor in self.factors]

    def probability(self, assignment):

        # Вычисляет вероятность конкретного набора значений переменных.
        
        # Аргументы probability:
        #    assignment (Dict[str, int]): Назначение значений переменным.
        
        # Возвращает:
        #    float: Вероятность данного набора значений.
        
        prob = 1.0
        for factor in self.factors:
            sub_assignment = tuple(assignment.get(var.name) for var in factor.vars)
            if None in sub_assignment:
                continue
            prob *= factor.table.get(sub_assignment, 0.0)
        return prob
        
    # Функции Julia (5.1)
    def bayesian_score_component(self, M, alpha):

        # Вычисляет компоненту байесовского критерия.
        
        # Аргументы bayesian_score_component:
        #    M (np.ndarray): Матрица частотных значений.
        #    alpha (np.ndarray): Апостериорные вероятности.
        
        # Возвращает:
        #    float: Значение компоненты байесовского критерия.
        
        M = np.atleast_2d(M)
        alpha = np.atleast_2d(alpha)
        p = np.sum(gammaln(alpha + M)) - np.sum(gammaln(alpha))
        p += np.sum(gammaln(np.sum(alpha, axis=1))) - np.sum(gammaln(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
        return p
        
    # Функции Julia (5.1)
    def bayesian_score(self, D):

        # Вычисляет байесовский балл сети на основании данных.
        
        # Аргументы bayesian_score:
        #    D (List[Dict[str, int]]): Данные для расчёта.
        
        # Возвращает:
        #    float: Байесовский балл.
        
        M = self.statistics(D)
        alpha = self.prior()
        return sum(self.bayesian_score_component(M[var], alpha[var]) for var in self.vars)

    # Метод (4.1)
    def statistics(self, D: List[Dict[str, int]]) -> Dict[str, np.ndarray]:

        # Подсчёт статистики частот появления значений переменных в данных.
        
        # Аргументы statistics:
        #    D (List[Dict[str, int]]): Список данных.
        
        # Возвращает:
        #    Dict[str, np.ndarray]: Матрицы частотных значений для каждой переменной.
        
        counts = {}
        for var_name, var in self.vars.items():
            parents = list(self.graph.predecessors(var_name))
            r_i = var.r
            q_i = int(np.prod([self.vars[p].r for p in parents])) if parents else 1
            M = np.zeros((q_i, r_i))
            for data_point in D:
                x_i = data_point[var_name] - 1
                parent_index = 0 if not parents else sum(data_point[p] - 1 for p in parents)
                M[parent_index, x_i] += 1
            counts[var_name] = M
        return counts

    # Метод prior (4.2)
    def prior(self) -> Dict[str, np.ndarray]:

        # Генерирует априорные вероятности для переменных сети.
        
        # Возвращает:
        #    Dict[str, np.ndarray]: Словарь априорных вероятностей для переменных.
        
        return {var.name: np.atleast_2d(np.ones((int(np.prod([self.vars[p].r for p in list(self.graph.predecessors(var.name))])), var.r)) * 5)
                if list(self.graph.predecessors(var.name)) else np.atleast_2d(np.ones(var.r) * 5)
                for var in self.vars.values()}

    # Функции Julia (5.1)
    
    def expected_utility(self, assignment):

        # Вычисляет ожидаемую полезность заданного набора значений переменных.
        
        # Аргументы expected_utility:
        #    assignment (Dict[str, int]): Значения переменных.
        
        # Возвращает:
        #    float: Ожидаемая полезность.
        
        utility = 0
        for var, table in self.utilities.items():
            sub_assignment = tuple(assignment.get(v) for v in self.vars if v in table)
            if None not in sub_assignment:
                utility += table.get(sub_assignment, 0)
        return utility

    # Метод (4.1)
    def sub2ind(self, siz, x):

        # Преобразует многомерные индексы в одномерный индекс.
        
        k = np.cumprod([1] + siz[:-1])
        return sum(x_i * k_i for x_i, k_i in zip(x, k))
    
    def plot_graph(self, title="Decision Network"):

        # Отображает граф Байесовской сети с различными типами узлов.
        
        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(self.graph)
        node_shapes = {"circle": "o", "square": "s", "diamond": "d"}
        node_types = {var: "circle" for var in self.vars}  # Default chance nodes
        for var in self.utilities:
            node_types[var] = "diamond"  # Utility nodes
        
        for node, shape in node_shapes.items():
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[n for n, t in node_types.items() if t == node], node_shape=shape, node_color='lightblue')
        
        nx.draw(self.graph, pos, with_labels=True, edge_color='gray', font_weight='bold')
        edge_labels = {(u, v): f"{u} → {v}" for u, v in self.graph.edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        plt.title(title)
        plt.show()