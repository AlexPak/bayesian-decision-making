import pandas as pd
import ast

class PandasFileOpen:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            # Display first few rows for debugging
            print("CSV Data Preview:")
            print(self.data.head())
            self.data.dropna(subset=['Variables', 'Values'], inplace=True)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None

    def to_factors(self):
        if self.data is None:
            raise ValueError("No data loaded")
    
        factors = []
        for _, row in self.data.iterrows():
            print(f"Processing row: {row}")  # Debug each row
            variables_str = row['Variables']
            values_str = row['Values']
            
            try:
                variables = ast.literal_eval(variables_str)
                values = ast.literal_eval(values_str)
                if isinstance(variables, list) and isinstance(values, dict):
                    factors.append((variables, values))
                else:
                    print(f"Skipping invalid row: {row}")
            except (ValueError, SyntaxError) as e:
                print(f"Skipping invalid row due to parsing error: {row} - {e}")
    
        return factors