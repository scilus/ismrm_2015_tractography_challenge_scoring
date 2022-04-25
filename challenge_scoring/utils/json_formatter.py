import json


def save_dict_to_json_file(path, dictionary):
    """ Saves a dict in a json formatted file. """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4,
                                   separators=(',', ': ')))


def load_dict_from_json_file(path):
    """ Loads a dict from a json formatted file. """
    with open(path, "r") as json_file:
        return json.loads(json_file.read())
