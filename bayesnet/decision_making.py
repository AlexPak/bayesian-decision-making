from typing import List, Dict, Tuple, Any
import numpy as np
import networkx as nx
from helpers.formulae import sub2ind

class Variable:
    def __init__(self, name: str, r: int):
        self.name: str = name # Имя переменной (например, 'x', 'y', 'z')
        self.r: int = r # Количество возможных значений для переменной
        self.values = list(range(1, r + 1)) # Возможные значения переменной

    def __str__(self):
        return f"Variable(name={self.name}, num_values={self.r})" #Вывод значений name и r(int) в текстовом формате

    def __repr__(self):
        return self.__str__() #Форматирование в строку значение __str__

    def __hash__(self):
        return hash(self.name) # Реализуется хэш-функция для использования переменных в качестве ключей

class Factor:
    def __init__(self, vars: List[Variable], table: Dict[Tuple[Variable, ...], float]): #Возможность добавления значение в table 
                        #(Как класса условной вероятности в параметре Variable, расчёт события в любом формате и его вероятности в формате float)
        self.vars = vars # Переменные, входящие в фактор
        self.table = table # Таблица вероятностей для различных комбинаций значений переменных

    def __str__(self):
        return f"Factor(vars={[str(var) for var in self.vars]}, table={self.table})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(tuple(self.vars))

    #Операция маргинализации удаляет переменную, суммируя по всем её значениям.
    
    def marginalization(self, var_name: str) -> 'Factor':
        new_vars = [var for var in self.vars if var.name != var_name] # Переменные, которые остаются после маргинализации
        new_table = {} # Новая таблица вероятностей
        for assignment, prob in self.table.items():
            new_assignment = tuple(
                value for var, value in zip(self.vars, assignment) if var.name != var_name #Запись решения на основе схожести
            )
            new_table[new_assignment] = new_table.get(new_assignment, 0) + prob #Суммирование всех вероятностей для исключаемой переменной
        return Factor(new_vars, new_table) #Запись фактора на основе вероятностей

    #Метод вычисления произведения двух факторов объединяет их переменные и перемножает вероятности.
    
    def product(self, other):
        # Объединяем переменные из обоих факторов, сохраняя порядок
        new_vars = list({var.name: var for var in self.vars + other.vars}.values())
        print("New Variables:", [var.name for var in new_vars])  # Debug: вывод переменных после объединения
        
        # Инициализируем новую таблицу вероятностей
        new_table = {}

        # Находим пересекающиеся переменные для сопоставления
        self_var_names = [var.name for var in self.vars]
        other_var_names = [var.name for var in other.vars]
        common_vars = set(self_var_names).intersection(other_var_names)
        print("Common Variables:", common_vars)  # Debug: вывод общих переменных
        
        # Проходим по всем возможным комбинациям значений переменных
        for assignment1, prob1 in self.table.items():
            for assignment2, prob2 in other.table.items():
                # Проверяем, что пересекающиеся переменные совпадают
                if all(assignment1[self_var_names.index(var)] == assignment2[other_var_names.index(var)]
                       for var in common_vars):
                    # Объединяем значения для новой таблицы
                    new_assignment = tuple(
                        assignment1[self_var_names.index(var)] if var in self_var_names else assignment2[other_var_names.index(var)]
                        for var in [v.name for v in new_vars]
                    )
                    # Считаем произведение вероятностей
                    new_table[new_assignment] = prob1 * prob2
                    print(f"Merging {assignment1} (prob={prob1}) and {assignment2} (prob={prob2}) -> {new_assignment} (prob={new_table[new_assignment]})")  # Debug
        return Factor(new_vars, new_table)

    #Нормализация обеспечивает, что сумма всех вероятностей в факторе будет равна 1.
    
    def normalize(self) -> 'Factor':
        total_prob = sum(self.table.values()) # Сумма всех вероятностей в таблице
        print("Total Probability before normalization:", total_prob)  # Debug
        
        if total_prob == 0:
            raise ValueError("Total probability is zero; normalization is impossible.")
            
        normalized_table = {assignment: prob / total_prob for assignment, prob in self.table.items()}

        for assignment, prob in normalized_table.items():
            print(f"Normalized assignment {assignment}: {prob}")  # Debug
        
        return Factor(self.vars, normalized_table)

    def condition(self, evidence: dict) -> 'Factor':
        """
        Conditions this factor on the given evidence, reducing the table accordingly.
        """
        new_table = {}
        for assignment, prob in self.table.items():
            matches = all(
                assignment[i] == evidence[var.name] for i, var in enumerate(self.vars) if var.name in evidence
            )
            
            if matches:
                # Формируем новое назначение без переменных из evidence
                new_assignment = tuple(
                    assignment[i] for i, var in enumerate(self.vars) if var.name not in evidence
                )
                new_table[new_assignment] = prob
    
        # Обновляем список переменных, исключая те, которые есть в evidence
        new_vars = [var for var in self.vars if var.name not in evidence]
        
        return Factor(new_vars, new_table)

class BayesianNetwork:
    def __init__(self, vars: List[Variable], factors: List[Factor], edges: List[Tuple[int, int]]):
        self.vars = vars
        self.factors = factors
        self.graph = nx.DiGraph(edges)

    def normalize_factors(self):
        """Normalizes each factor in the network."""
        for i in range(len(self.factors)):
            self.factors[i] = self.factors[i].normalize()

    def condition_factors(self, evidence):
        """Применяет условия для факторов на основе доказательств."""
        for i in range(len(self.factors)):
            self.factors[i] = self.factors[i].condition(evidence)
            print(f"Factor after conditioning with evidence {evidence}: {self.factors[i]}")

    def probability(self, assignment):
        """Вычисляет совместную вероятность для заданного назначения в сети."""
        prob = 1.0
        print("Initial Assignment:", assignment)
        
        for factor in self.factors:
            # Отбираем переменные назначения, которые соответствуют текущему фактору
            sub_assignment = tuple(assignment.get(var.name) for var in factor.vars)
            
            # Проверяем, что все значения в sub_assignment определены
            if None in sub_assignment:
                print(f"Skipping factor due to incomplete assignment for factor {factor}")
                continue
            
            # Если подназначение присутствует в таблице, берем его вероятность; иначе 0.0
            factor_prob = factor.table.get(sub_assignment, 0.0)
            
            # Отладка: отображение информации о факторе и вероятности
            print(f"Factor: {factor}")
            print(f"Sub-assignment for factor: {sub_assignment}")
            print(f"Probability from factor: {factor_prob}")
            
            prob *= factor_prob  # Умножаем на вероятность из фактора
        
        print("Final Computed Probability:", prob)
        return prob

    def statistics(self, D: np.ndarray) -> List[np.ndarray]:
        """Extracts count statistics for each variable given data D."""
        counts = []
        for i, var in enumerate(self.vars):
            parents = list(self.graph.predecessors(i))
            r_i = var.r
            q_i = int(np.prod([self.vars[p].r for p in parents]))  # Convert to integer
            M = np.zeros((q_i, r_i))
    
            for data_point in D:
                x_i = data_point[i] - 1  # Adjust index for zero-based array
                if parents:
                    parent_values = [data_point[p] - 1 for p in parents]
                    parent_index = sub2ind([self.vars[p].r for p in parents], parent_values)
                else:
                    parent_index = 0
                M[parent_index, x_i] += 1
            counts.append(M)
        return counts

    def prior(self) -> List[np.ndarray]:
        """Generates prior counts where all entries are initialized to 1."""
        prior_counts = []
        for i, var in enumerate(self.vars):
            parents = list(self.graph.predecessors(i))
            r_i = var.r
            q_i = int(np.prod([self.vars[p].r for p in parents]))  # Convert to integer
            prior_counts.append(np.ones((q_i, r_i)))
        return prior_counts

    def gaussian_kernel(self, b: float):
        """Returns a Gaussian kernel with bandwidth b."""
        return lambda x: (1 / (np.sqrt(2 * np.pi) * b)) * np.exp(-0.5 * (x / b)**2)

    def kernel_density_estimate(self, kernel, observations):
        """Estimates density with a kernel function over observations."""
        return lambda x: np.mean([kernel(x - o) for o in observations])