import logging
import os
import torch
import pickle
import json

from eisen_deploy.utils import json_file_to_dict
from eisen.utils import EisenModuleWrapper


logger = logging.getLogger(__name__)


class EisenServingHandler(object):
    """

    """

    def __init__(self):
        self.model = None
        self.device = None
        self.pre_process_tform = None
        self.post_process_tform = None
        self.metadata = None
        self.initialized = False
        self.input_name_list = []
        self.output_name_list = []

    def initialize(self, ctx):
        properties = ctx.system_properties

        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        model_dir = properties.get("model_dir")

        # Model file
        model_pt_path = os.path.join(model_dir, "model.pt")

        # Pre processing chain
        pre_processing_pkl = os.path.join(model_dir, "pre_process_tform.pkl")

        # Post processing chain
        post_processing_pkl = os.path.join(model_dir, "post_process_tform.pkl")

        # unpickle serialized transform chain
        with open(pre_processing_pkl, "rb") as f:
            self.pre_process_tform = pickle.load(f)

        with open(post_processing_pkl, "rb") as f:
            self.post_process_tform = pickle.load(f)

        # Metadata about the model
        metadata_json = os.path.join(model_dir, "metadata.json")

        self.metadata = json_file_to_dict(metadata_json)

        self.input_name_list = []
        for entry in self.metadata['inputs']:
            self.input_name_list.append(entry['name'])

        self.output_name_list = []
        for entry in self.metadata['outputs']:
            self.output_name_list.append(entry['name'])

        # deserialize pytorch model
        # todo check torchscript will work
        base_model = torch.load(model_pt_path, map_location=self.device)

        self.model = EisenModuleWrapper(base_model, self.input_name_list, self.output_name_list)

        # put model in eval mode
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))

        self.initialized = True

    def get_metadata(self):
        return [json.dumps(self.metadata)]

    def pre_process(self, data):
        """
        """

        input_dict = self.pre_process_tform(data)

        return input_dict

    def inference(self, input_dict, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        for name in self.model.input_names:
            input_dict[name] = torch.Tensor(input_dict[name]).to(self.device)

        output_dict = self.model(**input_dict)

        for name in self.model.output_names:
            output_dict[name] = output_dict[name].data.cpu().numpy()

        return output_dict

    def post_process(self, output_dict):
        prediction = self.post_process(output_dict)

        return prediction

    def handle(self, data):
        model_input = self.pre_process(data)
        model_out = self.inference(model_input)
        prediction = self.post_process(model_out)

        output_list = []

        for name in self.output_name_list:
            output_list.append(prediction[name])

        return output_list


_service = EisenServingHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None or data[0].keys() not in _service.input_name_list:
        return _service.get_metadata()

    else:
        return _service.handle(data[0])
