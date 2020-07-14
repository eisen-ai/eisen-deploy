import logging
import os
import torch
import pickle
import json

from eisen.utils import EisenModuleWrapper


logger = logging.getLogger(__name__)


def json_file_to_dict(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError('The JSON file {} cannot be read'.format(json_file))

    with open(json_file) as json_file:
        dictionary = json.load(json_file)

    return dictionary


class EisenServingHandler(object):
    """
    EisenServingHandler is a custom object to handle inference request within TorchServing. It is usually included
    automatically in the MAR.
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
        """
        Initializes the fields of the EisenServingHandler object based on the context.

        :param ctx: context of an inference request
        :return: None
        """
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

        self.input_name_list = self.metadata['model_input_list']

        self.output_name_list = self.metadata['model_output_list']

        # deserialize pytorch model
        base_model = torch.load(model_pt_path, map_location=self.device)

        self.model = EisenModuleWrapper(base_model, self.input_name_list, self.output_name_list)

        # put model in eval mode
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))

        self.initialized = True

    def get_metadata(self):
        """
        This function returns metadata about the model as JSON

        :return: list
        """
        return [json.dumps(self.metadata)]

    def pre_process(self, data):
        """
        Applies pre-processing transform using de-pickled transform chain in the MAR.

        :param data: dictionary containing a collated batch of data
        :type data: dict

        """
        input_dict = self.pre_process_tform(data)

        return input_dict

    def inference(self, input_dict):
        """
        Performs prediction using the model. Feeds the necessary information to the model starting from the
        received data and creates an output dictionary as a result.

        :param input_dict: input batch, in form of a dictionary of collated datapoints
        :type input_dict: dict

        :return: dict
        """

        for name in self.model.input_names:
            input_dict[name] = torch.Tensor(input_dict[name]).to(self.device)

        output_dict = self.model(**input_dict)

        for name in self.model.output_names:
            output_dict[name] = output_dict[name].data.cpu().numpy()

        return output_dict

    def post_process(self, output_dict):
        """
        Applies post-processing transform using de-pickled transform chain in the MAR.

        :param output_dict: dictionary containing the result of inference on a collated batch of data
        :type output_dict: dict
        """

        prediction = self.post_process_tform(output_dict)

        return prediction

    def handle(self, data):
        """
        Handles one request.

        :param data: dictionary of data
        :type data: dict

        :return: list
        """
        input_data = {}
        for input in self.metadata['inputs']:
            input_data[input['name']] = data[input['name']]
            
        model_input = self.pre_process(input_data)
        
        model_out = self.inference(model_input)

        model_out.update(model_input)  # output dictionary still contains inputs (which may be useful for tforms)

        prediction = self.post_process(model_out)
        
        output_data = {}
        for output in self.metadata['outputs']:
            output_data[output['name']] = prediction[output['name']]

        buffer = pickle.dumps(output_data)

        return [buffer]


_service = EisenServingHandler()


def handle(data, context):

    if not _service.initialized:
        _service.initialize(context)

    if data is not None and hasattr(data, '__getitem__') and 'body' in data[0].keys() and len(data[0]['body']) > 0:
        data = data[0]['body']
    else:
        return _service.get_metadata()

    data = pickle.loads(data)

    if not all([key in data.keys() for key in _service.input_name_list]):
        return _service.get_metadata()

    else:
        return _service.handle(data)
