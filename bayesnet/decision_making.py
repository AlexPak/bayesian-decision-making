from typing import List, Dict, Tuple, Any
import networkx as nx

class Variable:
    def __init__(self, name: str, r: int):
        self.name: str = name
        self.r: int = r
        self.values = list(range(1, r + 1))

    def __str__(self):
        return f"Variable(name={self.name}, num_values={self.r})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

class Factor:
    def __init__(self, vars: List[Variable], table: Dict[Tuple[Variable, ...], float]):
        self.vars = vars
        self.table = table

    def __str__(self):
        return f"Factor(vars={[str(var) for var in self.vars]}, table={self.table})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(tuple(self.vars))

    def marginalization(self, var_name: str) -> 'Factor':
        new_vars = [var for var in self.vars if var.name != var_name]
        new_table = {}
        for assignment, prob in self.table.items():
            new_assignment = tuple(value for var, value in zip(self.vars, assignment) if var.name != var_name)
            new_table[new_assignment] = new_table.get(new_assignment, 0) + prob
        return Factor(new_vars, new_table)

    def product(self, other: 'Factor') -> 'Factor':
        common_vars = [var for var in self.vars if var in other.vars]
        new_vars = list(self.vars) + [var for var in other.vars if var not in self.vars]
        new_table = {}
        for assignment1, prob1 in self.table.items():
            for assignment2, prob2 in other.table.items():
                if all(assignment1[self.vars.index(var)] == assignment2[other.vars.index(var)] for var in common_vars):
                    merged_assignment = tuple(
                        assignment1[self.vars.index(var)] if var in self.vars else assignment2[other.vars.index(var)]
                        for var in new_vars
                    )
                    new_table[merged_assignment] = prob1 * prob2
        return Factor(new_vars, new_table)

    def normalize(self) -> 'Factor':
        total_prob = sum(self.table.values())
        if total_prob == 0:
            raise ValueError("Total probability is zero; normalization is impossible.")
        normalized_table = {assignment: prob / total_prob for assignment, prob in self.table.items()}
        return Factor(self.vars, normalized_table)

    def condition(self, evidence: dict) -> 'Factor':
        new_table = {}
        for assignment, prob in self.table.items():
            matches = all(
                var.name in evidence and value == evidence[var.name]
                for var, value in zip(self.vars, assignment)
            )
            if matches:
                new_assignment = tuple(
                    value for var, value in zip(self.vars, assignment) if var.name not in evidence
                )
                new_table[new_assignment] = new_table.get(new_assignment, 0) + prob
        new_vars = [var for var in self.vars if var.name not in evidence]
        return Factor(new_vars, new_table)

class BayesianNetwork:
    def __init__(self, vars: List[Variable], factors: List[Factor], edges: List[Tuple[int, int]]):
        self.vars = vars
        self.factors = factors
        self.graph = nx.DiGraph(edges)

    def probability(self, assignment: Dict[str, Any]) -> float:
        prob = 1.0
        for factor in self.factors:
            sub_assignment = tuple(assignment[var.name] for var in factor.vars if var.name in assignment)
            prob *= factor.table.get(sub_assignment, 0.0)
        return prob

    def normalize_factors(self):
        for i in range(len(self.factors)):
            self.factors[i] = self.factors[i].normalize()

    def condition_factors(self, evidence: dict):
        for i in range(len(self.factors)):
            self.factors[i] = self.factors[i].condition(evidence)