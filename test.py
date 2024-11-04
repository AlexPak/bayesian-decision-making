import unittest
# import Variable
from bayesnet.decision_making import Factor, Variable, BayesianNetwork

class TestFactorMethods(unittest.TestCase):
    def test_marginalization(self):
        X0 = Variable('X',0)
        X1 = Variable('X',1)
        Y0 = Variable('Y',0)
        Y1 = Variable('Y',1)

        vars = [X0, X1, Y0, Y1]
        table = {
            (X0, Y0): 0.1,
            (X0, Y1): 0.9,
            (X1, Y0): 0.8,
            (X1, Y1): 0.2,
        }
        factor = Factor(vars, table)

        marginalized_factor = factor.marginalization('Y')

        expected_vars = [X0, X1]

        expected_table = {
            (X0): table[(X0,Y0)] + table[(X0,Y1)],  
            (X1): table[(X1,Y0)] + table[(X1,Y1)],  
        }

        print("marginalized_factor.table")
        print(marginalized_factor.table)
        print("expected_table")
        print(expected_table)

        self.assertEqual(marginalized_factor.vars, expected_vars)
        self.assertEqual(marginalized_factor.table, expected_table)

    def test_product(self):
        X0 = Variable('X', 0)
        X1 = Variable('X', 1)
        Y0 = Variable('Y', 0)
        Y1 = Variable('Y', 1)
        Z0 = Variable('Z', 0)
        Z1 = Variable('Z', 1)

        vars1 = [X0,X1,Y0,Y1]
        table1 = {
            (X0, Y0): 0.5,
            (X0, Y1): 0.8,
            (X1, Y0): 0.1,
            (X1, Y1): 0.4,
        }
        factor1 = Factor(vars1, table1)

        vars2 = [Y0,Y1,Z0,Z1]
        table2 = {
            (Y0, Z0): 0.2,
            (Y0, Z1): 0.6,
            (Y1, Z0): 0.3,
            (Y1, Z1): 0.7,
        }
        factor2 = Factor(vars2, table2)

        product_factor = factor1.product(factor2)

        expected_vars = [X0,X1,Y0,Y1,Z0,Z1]

        expected_table = {
            (X0, Y0, Z0): table1[(X0,Y0)] * table2[(Y0,Z0)],
            (X0, Y0, Z1): table1[(X0,Y0)] * table2[(Y0,Z1)],
            (X0, Y1, Z0): table1[(X0,Y1)] * table2[(Y1,Z0)],
            (X0, Y1, Z1): table1[(X0,Y1)] * table2[(Y0,Z1)],
            (X1, Y0, Z0): table1[(X1,Y0)] * table2[(Y0,Z0)],
            (X1, Y0, Z1): table1[(X1,Y0)] * table2[(Y0,Z1)],
            (X1, Y1, Z0): table1[(X1,Y1)] * table2[(Y1,Z0)],
            (X1, Y1, Z1): table1[(X1,Y1)] * table2[(Y0,Z1)],
        }

        for assignment, expected_prob in expected_table.items():
            result_prob = product_factor.table.get(assignment, None)
            self.assertIsNotNone(result_prob, f"Assignment {assignment} missing in result table.")
            self.assertAlmostEqual(result_prob, expected_prob, places=7, msg=f"Probability mismatch for assignment {assignment}")

        self.assertEqual(set(product_factor.table.keys()), set(expected_table.keys()))

        self.assertEqual(set(var.name for var in product_factor.vars), set(var.name for var in expected_vars))

if __name__ == '__main__':
    unittest.main()
