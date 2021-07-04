import json, os


def dir_exist(dir_path):
    return os.path.isdir(dir_path)


def create_path(path):
    os.makedirs(path)


def get_files(path):
    return os.listdir(path)


def get_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data