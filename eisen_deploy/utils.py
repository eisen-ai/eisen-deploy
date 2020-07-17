import numpy as np


def encode_data(obj):
    """
    Encodes data before serialization. Basic data-types that can be directly serialized are
    returned unchanged. Other data types that cannot be serialized by msgpack, are broken down
    and made compatible as well as recognizable. Numpy array are a perfect example of this need.
    As they are not serializable "as-is", they are transferred in a broken down format.

    If the passed object is of a type not supported by this function,
    the object is returned unchanged.

    This function is called recursively on dictionaries stored within dictionary elements.
    """
    if isinstance(obj, (int, float, list, bool, str)):
        return obj

    elif isinstance(obj, dict):
        for key in obj.keys():
            obj[key] = encode_data(obj[key])

        return obj

    elif isinstance(obj, np.ndarray):
        return {
            '__ndarray__': True,
            'data': obj.tostring(),
            'shape': obj.shape,
            'type': str(obj.dtype)
        }

    return obj


def decode_data(obj):
    """
    Decodes data that has been previously encoded by encode_data (see above).

    Data is restored to its original format and type by recognizing what kind of transformation
    has been previously done by encode_data.

    Data types that cannot be decoded are returned unchanged.
    """
    if isinstance(obj, (int, float, list, bool, str)):
        return obj

    elif '__ndarray__' in obj:
        return np.frombuffer(obj['data'], dtype=obj['type']).reshape(*obj['shape'])

    elif isinstance(obj, dict):
        for key in obj.keys():
            obj[key] = decode_data(obj[key])

        return obj

    return obj