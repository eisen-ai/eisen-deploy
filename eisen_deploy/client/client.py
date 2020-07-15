import requests
import pickle
import numpy as np

from pickle import UnpicklingError


class EisenServingClient:
    """
    Eisen Serving client functionality. This object implements communication with prediction models packaged
    via EisenServingMAR. This client makes the assumption that EisenServing handler is used within the MAR.

    .. code-block:: python

            from eisen_deploy.client import EisenServingClient

            client = EisenServingClient(url='http://localhost/...')

            metadata = client.get_metadata()

            output = client.predict(input_data)

    """
    def __init__(self, url, validate_inputs=False):
        """
        Initializes the client object.

        .. code-block:: python

            from eisen_deploy.client import EisenServingClient

            client = EisenServingClient(url='http://localhost/...')

        :param url: url of prediction endpoint
        :type url: str
        :param validate_inputs: Whether inputs should be validated in terms of shape and type before sending them
        :type validate_inputs: bool
        """
        self.url = url
        self.validate_inputs = validate_inputs

        self.metadata = self.get_metadata()

    def input_validation(self, batch):
        assert len(batch.keys()) == len(self.metadata['inputs'])

        for input in self.metadata['inputs']:

            key = input['name']

            if input['type'] == 'ndarray':
                desired_ndim = len(input['shape'])

                valid_desired_shape = np.asarray(input['shape'])
                invalid_shape_dims = (valid_desired_shape != -1)
                valid_desired_shape = valid_desired_shape[invalid_shape_dims]

                valid_input_shape = np.asarray(batch[key].shape)
                valid_input_shape = valid_input_shape[invalid_shape_dims]

                assert batch[key].ndim == desired_ndim
                assert np.prod(valid_desired_shape) == np.prod(valid_input_shape)

            elif input['type'] == 'list' or input['type'] == 'str':
                assert len(batch[key]) == input['shape']

    def get_metadata(self):
        """
        Get model metadata as a result of an empty query to the model. This allows to receive information
        about the model (Eg. its inputs and outputs).

        :return: dict
        """
        response = requests.post(url=self.url)

        return response.json()

    def predict(self, batch):
        """
        Predict a data batch using the remote model deployed via EisenServing

        :param batch: dictionary representing a collated batch of data to put as input to the neural network
        :type batch: dict

        :return: dict
        """
        if self.validate_inputs:
            self.input_validation(batch)

        buffer = pickle.dumps(batch)

        response = requests.post(url=self.url, data=buffer)

        try:
            prediction = pickle.loads(response.content)

        except UnpicklingError:
            print('There was an error during your request. The server has responded in an unexpected way.')

            return response.json()

        return prediction
