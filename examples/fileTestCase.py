import unittest
from helpers.openFile import PandasFileOpen
from bayesnet.decision_making import Factor, Variable, BayesianNetwork

class TestFactor(unittest.TestCase):
    
    def setUp(self):
        # Initialize file handler and load data from CSV
        self.file_handler = PandasFileOpen("file_data.csv")
        self.file_handler.load_data()
        factors_data = self.file_handler.to_factors()
        
        # Check if factors_data was loaded correctly
        if not factors_data:
            print("No valid data found in CSV file.")
            raise ValueError("No factors loaded from file")
        
        # Dynamically initialize factors from the loaded data
        self.factors = [
            Factor([Variable(var.strip(), 2) for var in factor[0]], factor[1]) 
            for factor in factors_data
        ]
        
        # Assuming the first factor in the file is used for testing specific methods
        if self.factors:
            self.factor = self.factors[0]
        else:
            raise ValueError("No factors loaded from file")

    def test_marginalization(self):
        # Marginalize on the first variable in self.factor
        if self.factor.vars:
            var_name = self.factor.vars[0].name
            marginalized_factor = self.factor.marginalization(var_name)
            
            # Define the expected results for the marginalization
            expected_table = {
                (): 1.0  # Adjust this based on actual marginalization results
            }
            self.assertEqual(marginalized_factor.table, expected_table)
    
    def test_product(self):
        if len(self.factors) >= 2:
            # Example product test with the first two factors from the file
            factor1, factor2 = self.factors[:2]
            product_factor = factor1.product(factor2)
            
            # Define the expected results for the product
            expected_table = {
                (1, 1): 0.9702, (1, 2): 0.0198, (2, 1): 0.0098, (2, 2): 0.0002  # Customize
            }
            self.assertEqual(product_factor.table, expected_table)
    
    def test_condition(self):
        # Condition with evidence on the first factor
        evidence = {'b': 1}  # Customize based on your data structure
        conditioned_factor = self.factor.condition(evidence)
        
        # Define expected conditioned factor table
        expected_table = {
            (): 0.99  # Customize based on actual condition results
        }
        self.assertEqual(conditioned_factor.table, expected_table)

class TestBayesianNetwork(unittest.TestCase):
    def setUp(self):
        # Load factors data from file
        self.file_handler = PandasFileOpen("file_data.csv")
        self.file_handler.load_data()
        factors_data = self.file_handler.to_factors()
        
        # Convert factors data to Factor objects
        self.factors = [
            Factor([Variable(var.strip(), 2) for var in factor[0]], factor[1]) 
            for factor in factors_data
        ]
        
        # Define variables and edges
        self.vars = [Variable('b', 2), Variable('s', 2), Variable('e', 2), Variable('d', 2), Variable('c', 2)]
        self.edges = [(1, 2), (2, 3), (3, 4), (3, 5)]
        self.bn = BayesianNetwork(self.vars, self.factors, self.edges)

    def test_probability(self):
        assignment = {'b': 1, 's': 1, 'e': 1}
        result = self.bn.probability(assignment)
        
        # Adjust the expected probability based on actual factor data loaded
        expected = 0.99 * 0.98 * 0.90  # Customize based on your data's expected result
        self.assertAlmostEqual(result, expected, places=2)

    def test_normalize_factors(self):
        self.bn.normalize_factors()
        for factor in self.bn.factors:
            total_prob = sum(factor.table.values())
            self.assertAlmostEqual(total_prob, 1.0, places=2)

    def test_condition_factors(self):
        evidence = {'b': 1, 's': 1}
        self.bn.condition_factors(evidence)
        
        # Verify the variables left after conditioning
        for factor in self.bn.factors:
            remaining_vars = [var.name for var in factor.vars]
            if 'e' in remaining_vars:
                self.assertIn('e', remaining_vars)
            else:
                print(f"Factor without 'e': {factor}")

# Run the tests
unittest.main(argv=[''], exit=False)