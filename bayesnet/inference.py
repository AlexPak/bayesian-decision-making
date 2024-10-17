import hashlib
import sys
from typing import List

def hash_parameters_and_result(params: str, result: float) -> str:
    hash_input = f"Params: {params}, Result: {result}"
    hash_object = hashlib.sha256(hash_input.encode())
    return hash_object.hexdigest()

def calculate_memory_usage(variables: List, factors: List) -> int:
    memory_usage = sum(sys.getsizeof(var) for var in variables)
    memory_usage += sum(sys.getsizeof(factor) + sys.getsizeof(factor.table) for factor in factors)
    return memory_usage