import psutil

def track_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss

def display_memory_usage(before: int, after: int):
    used_memory = after - before
    print(f"Использование памяти до: {before / (1024 ** 2):.2f} MB")
    print(f"Использование памяти после: {after / (1024 ** 2):.2f} MB")
    print(f"Разница в использовании памяти: {used_memory / (1024 ** 2):.2f} MB")