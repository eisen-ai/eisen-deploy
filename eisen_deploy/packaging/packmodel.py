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


class TorchServeMAR:
    """
        This object implements model packaging compliant with PyTorch serving. This kind of packaging
        is referred will generate a MAR package. This follows the PyTorch standard, which has been documented
        here https://github.com/pytorch/serve/blob/master/README.md#serve-a-model

        Once the model is packaged it can be used for inference via TorchServe.

        Saving a MAR package for a model requires an Eisen pre-processing transform object,
        Eisen post-processing transform object, a model object (torch.nn.Module) and a metadata dictionary.

        These components will be serialized and included in the MAR.

        The default request handler for TorchServe is eisen_deploy.serving.EisenServingHandler. This parameter can be
        overridden by specifying the path of a custom handler or using one of the custom handlers provided by PyTorch.
        When the default handler is overridden, the pre- and post- processing transforms as well as the metadata
        might be ignored and the behavior during serving might differ from expected.

        .. code-block:: python

            from eisen_deploy.packaging import TorchServeMAR

            my_model =  # Eg. A torch.nn.Module instance


            my_pre_processing =  # Eg. A pre processing transform object

            my_post_processing =  # Eg. A pre processing transform object

            metadata = {'inputs': [], 'outputs': []}  # metadata dictionary


            mar_creator = TorchServeMAR(my_pre_processing, my_post_processing, metadata)

            mar_creator(my_model, '/path/to/archive')

    """
    def __init__(self, pre_processing, post_processing, meta_data, handler=EISEN_SERVING_HANDLER_PATH):
        self.tmp_dir = tempfile.mkdtemp()

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

    def __call__(self, model, dst_path, model_name='model', model_version='test'):
        # save model
        torch.save(model, os.path.join(self.tmp_dir, 'model.pt'))

        args = ArgClass(
            os.path.join(self.tmp_dir, 'model.pt'),
            self.handler,
            ','.join([
                os.path.join(self.tmp_dir, 'pre_process_tform.pkl'),
                os.path.join(self.tmp_dir, 'post_process_tform.pkl'),
                os.path.join(self.tmp_dir, 'metadata.json')
            ]),
            dst_path,
            model_name,
            model_version
        )

        manifest = ModelExportUtils.generate_manifest_json(args)

        package_model(args, manifest=manifest)
