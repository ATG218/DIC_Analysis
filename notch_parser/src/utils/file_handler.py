def create_directory(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file(source, destination):
    import shutil
    shutil.copy2(source, destination)

def read_timestamp_from_file(file_path):
    with open(file_path, 'r') as file:
        timestamp = file.read().strip()
    return timestamp