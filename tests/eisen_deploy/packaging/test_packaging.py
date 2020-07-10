import torch
import tempfile
import os
import shutil
import pickle
import numpy as np
import json

from eisen_deploy.packaging import EisenServingMAR
from eisen_deploy.packaging import create_metadata


class IdentityTForm:
    def __call__(self, data):
        return data


class TestEisenServingMAR:
    def setup_class(self):
        self.model = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=0)
        self.pre_processing = IdentityTForm()
        self.post_processing = IdentityTForm()

        self.metadata = {
            'inputs': [{'name': 'input', 'type': 'np.ndarray', 'shape': [-1]}],
            'outputs': [{'name': 'output', 'type': 'np.ndarray', 'shape': [-1]}],
            'model_input_list': ['input'],
            'model_output_list': ['output'],
            'custom': {}
        }

        self.tmp_path = tempfile.mkdtemp()

    def test_pack_model(self):
        packer = EisenServingMAR(
            pre_processing=self.pre_processing,
            post_processing=self.post_processing,
            meta_data=self.metadata
        )

        dst_path = os.path.join(self.tmp_path, 'test_pack_model')

        os.mkdir(dst_path)

        packer.pack(self.model, dst_path, 'test_model', '1.0')

        assert os.path.exists(os.path.join(dst_path, 'test_model.mar'))

        unpack_mar_path = os.path.join(dst_path, 'unpack')

        os.makedirs(unpack_mar_path, exist_ok=True)

        shutil.unpack_archive(os.path.join(dst_path, 'test_model.mar'), unpack_mar_path, "zip")

        with open(os.path.join(unpack_mar_path, "pre_process_tform.pkl"), "rb") as f:
            pre_process_tform = pickle.load(f)

        with open(os.path.join(unpack_mar_path, "post_process_tform.pkl"), "rb") as f:
            post_process_tform = pickle.load(f)

        assert isinstance(pre_process_tform, IdentityTForm)

        assert isinstance(post_process_tform, IdentityTForm)

        model = torch.load(os.path.join(unpack_mar_path, "model.pt"))

        input = torch.Tensor(np.random.randn(1, 1, 3).astype(np.float32)).cpu()

        output_mod = model(input)

        output_ref = self.model(input)

        assert np.all(output_mod.data.cpu().numpy() == output_ref.data.cpu().numpy())

        with open(os.path.join(unpack_mar_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        assert metadata == self.metadata

    def test_create_metadata(self):
        metadata = create_metadata(
            input_name_list=['input'],
            input_type_list=['np.ndarray'],
            input_shape_list=[[-1]],
            output_name_list=['output'],
            output_type_list=['np.ndarray'],
            output_shape_list=[[-1]]
        )

        assert metadata == self.metadata

        metadata = create_metadata(
            input_name_list=['input'],
            input_type_list=['np.ndarray'],
            input_shape_list=[[-1]],
            output_name_list=['output'],
            output_type_list=['np.ndarray'],
            output_shape_list=[[-1]],
            model_input_list=['input'],
            model_output_list=['output']
        )

        assert metadata == self.metadata

        complex_metadada = {
            'inputs': [
                {'name': 'input1', 'type': 'np.ndarray', 'shape': [-1]},
                {'name': 'input2', 'type': 'np.ndarray', 'shape': [-1, 5, 14, 15, 16]}
            ],
            'outputs': [
                {'name': 'output1', 'type': 'json', 'shape': [-1]},
                {'name': 'output2', 'type': 'np.ndarray', 'shape': [-1, 3, 5]}
            ],
            'model_input_list': ['input1'],
            'model_output_list': ['output1'],
            'custom': {
                'test': 't',
                'test2': 't2'
            }
        }

        metadata = create_metadata(
            input_name_list=['input1', 'input2'],
            input_type_list=['np.ndarray', 'np.ndarray'],
            input_shape_list=[[-1], [-1, 5, 14, 15, 16]],
            output_name_list=['output1', 'output2'],
            output_type_list=['json', 'np.ndarray'],
            output_shape_list=[[-1], [-1, 3, 5]],
            model_input_list=['input1'],
            model_output_list=['output1'],
            custom_meta_dict={'test': 't', 'test2': 't2'}
        )

        assert complex_metadada == metadata

    def teardown_class(self):
        shutil.rmtree(self.tmp_path)