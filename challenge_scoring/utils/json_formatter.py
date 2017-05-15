import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}

        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        return np.array(dct['__ndarray__'], dtype=np.float32)

    return dct


def save_dict_to_json_file(path, dictionary):
    """ Saves a dict in a json formatted file. """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': '), cls=NumpyEncoder))


def load_dict_from_json_file(path):
    """ Loads a dict from a json formatted file. """
    with open(path, "r") as json_file:
        return json.loads(json_file.read(), object_hook=json_numpy_obj_hook)
