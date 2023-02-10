import yaml


def read_yaml(file_path: str):
    try:
        with open(file_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise e
