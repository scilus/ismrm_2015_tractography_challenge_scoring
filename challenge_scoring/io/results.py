from challenge_scoring.utils import json_formatter


def save_results(path, results):
    """ Save results in JSON file """
    json_formatter.save_dict_to_json_file(path, results)


def load_results(path):
    """ Load results from a JSON file """
    json_formatter.load_dict_from_json_file(path)
