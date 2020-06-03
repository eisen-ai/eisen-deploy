import requests
import pickle


class EisenServingClient:
    def __init__(self, url):
        self.url = url

    def get_metadata(self):
        response = requests.post(url=self.url)

        return response.json()

    def predict(self, batch):
        buffer = pickle.dumps(batch)

        response = requests.post(url=self.url, data=buffer)

        prediction = pickle.loads(response.content)

        return prediction