import requests
import pickle


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
    def __init__(self, url):
        """
        Initializes the client object.

        .. code-block:: python

            from eisen_deploy.client import EisenServingClient

            client = EisenServingClient(url='http://localhost/...')

        :param url: url of prediction endpoint
        :type url: str
        """
        self.url = url

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
        buffer = pickle.dumps(batch)

        response = requests.post(url=self.url, data=buffer)

        prediction = pickle.loads(response.content)

        return prediction