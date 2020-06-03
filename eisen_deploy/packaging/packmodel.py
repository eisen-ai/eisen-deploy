import tempfile
import shutil
import torch
import os
import pickle
import json
import inspect

from eisen_deploy.serving import EisenServingHandler

from model_archiver.manifest_components.manifest import RuntimeType
from model_archiver.model_packaging_utils import ModelExportUtils
from model_archiver.model_packaging import package_model


EISEN_SERVING_HANDLER_PATH = os.path.dirname(
    os.path.join(os.path.abspath(inspect.getsourcefile(EisenServingHandler)), 'handlers.py')
)


class ArgClass:
    """
    This is just a mock of arguments that would need to be otherwise passed via command line interface.
    """
    def __init__(self, serialized_file, handler, extra_files, export_path, model_name='model', model_version=None):
        self.model_name = model_name
        self.serialized_file = serialized_file
        self.handler = handler
        self.extra_files = extra_files
        self.export_path = export_path
        self.runtime = RuntimeType.PYTHON.value
        self.archive_format = 'default'
        self.version = model_version
        self.model_file = None
        self.source_vocab = None
        self.force = False


class EisenServingMAR:
    """
        This object implements model packaging compliant with PyTorch serving. This kind of packaging
        is referred as a MAR package. It follows the PyTorch standard, which is documented
        here https://github.com/pytorch/serve/tree/master/model-archiver

        Once the model is packaged it can be used for inference via TorchServe. Packing the model will
        in fact result in a <filename>.mar package (which usually is a .tar.gz archive) that can be used
        through the following command:

        .. code-block:: console

            torchserve --start --ncs --model-store model_zoo --models model.mar

    """
    def __init__(self, pre_processing, post_processing, meta_data, handler=None):
        """
        Saving a MAR package for a model requires an Eisen pre-processing transform object,
        Eisen post-processing transform object, a model object (torch.nn.Module) and a metadata dictionary.

        These components will be serialized and included in the MAR.

        The default request handler for TorchServe is eisen_deploy.serving.EisenServingHandler. This parameter can be
        overridden by specifying the path of a custom handler or using one of the custom handlers provided by PyTorch.
        When the default handler is overridden, the pre- and post- processing transforms as well as the metadata
        might be ignored and the behavior during serving might differ from expected.

        .. code-block:: python

            from eisen_deploy.packaging import EisenServingMAR

            my_model =  # Eg. A torch.nn.Module instance


            my_pre_processing =  # Eg. A pre processing transform object

            my_post_processing =  # Eg. A pre processing transform object

            metadata = {'inputs': [], 'outputs': []}  # metadata dictionary


            mar_creator = EisenServingMAR(my_pre_processing, my_post_processing, metadata)

            mar_creator.pack(my_model, '/path/to/archive')


        :param pre_processing: pre processing transform object. Will be pickled into a pickle file
        :type pre_processing: callable
        :param post_processing: post processing transform object. Will be pickled into a pickle file
        :type post_processing: callable
        :param meta_data: dictionary containing meta data about the model (Eg. information about inputs and outputs)
        :type meta_data: dict
        :param handler: name or filename of the handler. It is an optional parameter which rarely needs to be changed
        :type handler: str

        """
        self.tmp_dir = tempfile.mkdtemp()

        if handler is None:
            handler = EISEN_SERVING_HANDLER_PATH

        # save transform chain
        with open(os.path.join(self.tmp_dir, 'pre_process_tform.pkl'), 'wb') as f:
            pickle.dump(pre_processing, f)

        # save transform chain
        with open(os.path.join(self.tmp_dir, 'post_process_tform.pkl'), 'wb') as f:
            pickle.dump(post_processing, f)

        # save metadata
        with open(os.path.join(self.tmp_dir, 'metadata.json'), "w") as f:
            json.dump(meta_data, f)

        self.handler = handler

    def __del__(self):
        shutil.rmtree(self.tmp_dir)

    def pack(self, model, dst_path, model_name, model_version, additional_files=None):
        """
        Package a model into the MAR archive so that it can be used for serving using TorchServing

        :param model: an object representing a model
        :type model: torch.nn.Module
        :param dst_path: the destination base path (do not include the filename) of the MAR
        :type dst_path: str
        :param model_name: the name of the model (will be also used to define the prediction endpoint)
        :type model_name: str
        :param model_version: a string encoding the version of the model
        :type model_version: str
        :param additional_files: an optional list of files that should be included in the MAR
        :type additional_files: iterable

        :return: None
        """
        if additional_files is None:
            additional_files = []

        # save model
        torch.save(model, os.path.join(self.tmp_dir, 'model.pt'))

        args = ArgClass(
            os.path.join(self.tmp_dir, 'model.pt'),
            self.handler,
            ','.join([
                os.path.join(self.tmp_dir, 'pre_process_tform.pkl'),
                os.path.join(self.tmp_dir, 'post_process_tform.pkl'),
                os.path.join(self.tmp_dir, 'metadata.json')
            ] + additional_files),
            dst_path,
            model_name,
            model_version
        )

        manifest = ModelExportUtils.generate_manifest_json(args)

        package_model(args, manifest=manifest)
