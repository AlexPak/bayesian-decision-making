from typing import List
from decision_making import Variable, Factor
import ast
import pandas as pd

class PandasFileOpen:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("CSV Data Preview:")
            print(self.data.head())
            self.data.dropna(subset=['Variables', 'Values'], inplace=True)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None

    def to_factors(self) -> List[Factor]:
        if self.data is None:
            raise ValueError("No data loaded")
    
        factors = []
        for _, row in self.data.iterrows():
            try:
                variables = [Variable(var['name'], var['r']) for var in ast.literal_eval(row['Variables'])]
                values = ast.literal_eval(row['Values'])
                if isinstance(variables, list) and isinstance(values, dict):
                    factors.append(Factor(variables, values))
                else:
                    print(f"Skipping invalid row: {row}")
            except (ValueError, SyntaxError) as e:
                print(f"Skipping invalid row due to parsing error: {row} - {e}")
        
        return factors