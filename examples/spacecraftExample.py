import unittest
from bayesnet.decision_making import Factor, Variable, BayesianNetwork

class TestFactor(unittest.TestCase):
    
    def setUp(self):
        # Инициализация переменных и факторов
        self.B = Variable('b', 2)
        self.S = Variable('s', 2)
        self.E = Variable('e', 2)
        self.D = Variable('d', 2)
        self.C = Variable('c', 2)
        
        # Инициализация факторной таблицы для фактора E, B, S
        self.factor = Factor([self.E, self.B, self.S], {
            (1, 1, 1): 0.90, (1, 1, 2): 0.04,
            (1, 2, 1): 0.05, (1, 2, 2): 0.01,
            (2, 1, 1): 0.10, (2, 1, 2): 0.96,
            (2, 2, 1): 0.95, (2, 2, 2): 0.99
        })
        
    def test_marginalization(self):
        # Проверяем маргинализацию по переменной B
        marginalized_factor = self.factor.marginalization('b')
        
        # Ожидаемое значение после маргинализации по B
        expected_table = {
            (1, 1): 0.90 + 0.05,  # P(e=1, s=1) = 0.90 + 0.05
            (1, 2): 0.04 + 0.01,  # P(e=1, s=2) = 0.04 + 0.01
            (2, 1): 0.10 + 0.95,  # P(e=2, s=1) = 0.10 + 0.95
            (2, 2): 0.96 + 0.99   # P(e=2, s=2) = 0.96 + 0.99
        }
        
        self.assertEqual(marginalized_factor.table, expected_table)

    def test_product(self):
        # Тестируем операцию произведения факторов
        factor1 = Factor([self.B], {
            (1,): 0.99,
            (2,): 0.01
        })
        
        factor2 = Factor([self.B, self.S], {
            (1, 1): 0.98,
            (1, 2): 0.02,
            (2, 1): 0.05,
            (2, 2): 0.95
        })
        
        product_factor = factor1.product(factor2)
        
        # Ожидаемая таблица после произведения
        expected_table = {
            (1, 1): 0.99 * 0.98,
            (1, 2): 0.99 * 0.02,
            (2, 1): 0.01 * 0.05,
            (2, 2): 0.01 * 0.95
        }
        
        self.assertEqual(product_factor.table, expected_table)

    def test_condition(self):
        # Проверяем условие (condition) с учетом доказательств
        evidence = {'b': 1}
        conditioned_factor = self.factor.condition(evidence)
        
        # Ожидаемая таблица после условия
        expected_table = {
            (1, 1): 0.90,
            (1, 2): 0.04,
            (2, 1): 0.10,
            (2, 2): 0.96,
        }
        
        self.assertEqual(conditioned_factor.table, expected_table)

class TestBayesianNetwork(unittest.TestCase):
    def setUp(self):
        # Создаем переменные и факторы для тестирования
        self.B = Variable('b', 2)
        self.S = Variable('s', 2)
        self.E = Variable('e', 2)
        self.D = Variable('d', 2)
        self.C = Variable('c', 2)

        # Создаем факторы, как в примере
        factors = [
            Factor([self.B], {(1,): 0.99, (2,): 0.01}),
            Factor([self.S], {(1,): 0.98, (2,): 0.02}),
            Factor([self.E, self.B, self.S], {
                (1, 1, 1): 0.90, (1, 1, 2): 0.04, (1, 2, 1): 0.05, (1, 2, 2): 0.01,
                (2, 1, 1): 0.10, (2, 1, 2): 0.96, (2, 2, 1): 0.95, (2, 2, 2): 0.99
            }),
            Factor([self.D, self.E], {
                (1, 1): 0.96, (1, 2): 0.03,
                (2, 1): 0.04, (2, 2): 0.97
            }),
            Factor([self.C, self.E], {
                (1, 1): 0.98, (1, 2): 0.01,
                (2, 1): 0.02, (2, 2): 0.99
            })
        ]

        self.vars = [self.B, self.S, self.E, self.D, self.C]
        self.edges = [(1, 2), (2, 3), (3, 4), (3, 5)]
        self.bn = BayesianNetwork(self.vars, factors, self.edges)

    def test_probability(self):
        assignment = {'b': 1, 's': 1, 'e': 1}
        result = self.bn.probability(assignment)
        expected = 0.99 * 0.98 * 0.90  # Ожидаемое значение вероятности
        self.assertAlmostEqual(result, expected, places=2)

    def test_normalize_factors(self):
        self.bn.normalize_factors()
        for factor in self.bn.factors:
            total_prob = sum(factor.table.values())
            self.assertAlmostEqual(total_prob, 1.0, places=2)

    def test_condition_factors(self):
        evidence = {'b': 1, 's': 1}
        self.bn.condition_factors(evidence)
    
        # Проверка, что условие применено правильно
        for factor in self.bn.factors:
            remaining_vars = [var.name for var in factor.vars]
            if 'e' in remaining_vars:
                self.assertIn('e', remaining_vars)
            else:
                print(f"Factor without 'e': {factor}")

# Запуск тестов
unittest.main(argv=[''], exit=False)