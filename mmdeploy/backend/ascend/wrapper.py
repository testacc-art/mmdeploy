# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, NamedTuple, Sequence

import acl
import numpy as np
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper

_from_acl_data_type = {0: np.float32, 3: np.int32, 9: np.int64}


class AclError(Exception):
    """Acl Exception."""
    pass


def _check(code: int, msg: str):
    """check the error code.

    Args:
        code (int): The error code.
        msg (str): Error message.
    """
    if code != 0:
        raise AclError(msg, code)


class DataBuffer:
    """The acl data buffer.

    Args:
        size (int): Buffer size.
    """

    def __init__(self, size: int):
        data, ret = acl.rt.malloc(size, 0)
        _check(ret, 'acl.rt.malloc')
        self.data = data
        self.size = size
        self.handle = acl.create_data_buffer(data, size)

    def __del__(self):
        acl.destroy_data_buffer(self.handle)
        acl.rt.free(self.data)


class Dataset:
    """The acl dataset."""

    def __init__(self):
        self.handle = acl.mdl.create_dataset()
        self.buffers = []

    def __del__(self):
        acl.mdl.destroy_dataset(self.handle)

    def add_buffer(self, buffer: DataBuffer):
        """Add data buffer into the dataset.

        Args:
            buffer (DataBuffer): The DataBuffer instance.
        """
        self.buffers.append(buffer)
        _, ret = acl.mdl.add_dataset_buffer(self.handle, buffer.handle)
        _check(ret, 'acl.mdl.add_dataset_buffer')


class Binding(NamedTuple):
    index: int
    name: str
    dims: List[int]
    data_type: np.dtype
    size: int


class ModelDesc:
    """The model description wrapper.

    Args:
        model_id (int): The id of the model, created by acl tools.
    """

    def __init__(self, model_id):
        self._desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._desc, model_id)
        _check(ret, 'acl.mdl.get_desc')

        self.inputs = []
        self.dynamic_tensor = None
        num_inputs = acl.mdl.get_num_inputs(self._desc)
        for index in range(num_inputs):
            dims = self._get_input_dims(index)
            data_type = acl.mdl.get_input_data_type(self._desc, index)
            data_type = _from_acl_data_type[data_type]
            size = acl.mdl.get_input_size_by_index(self._desc, index)
            binding = Binding(index, dims['name'], dims['dims'], data_type,
                              size)
            if dims['name'] == 'ascend_mbatch_shape_data':
                self.dynamic_tensor = binding
            else:
                self.inputs.append(binding)

        self.outputs = []
        num_outputs = acl.mdl.get_num_outputs(self._desc)
        for index in range(num_outputs):
            dims = self._get_output_dims(index)
            data_type = acl.mdl.get_output_data_type(self._desc, index)
            data_type = _from_acl_data_type[data_type]
            size = acl.mdl.get_output_size_by_index(self._desc, index)
            self.outputs.append(
                Binding(index, dims['name'], dims['dims'], data_type, size))

    def __del__(self):
        acl.mdl.destroy_desc(self._desc)

    def _get_input_dims(self, index: int):
        """Get the dimension of the input by index.

        Args:
            index (int): The index of the input.
        """
        dims, ret = acl.mdl.get_input_dims(self._desc, index)
        _check(ret, 'acl.mdl.get_input_dims')
        return dims

    def _get_output_dims(self, index: int):
        """Get the dimension of the output by index.

        Args:
            index (int): The index of the output.
        """
        dims, ret = acl.mdl.get_output_dims(self._desc, index)
        _check(ret, 'acl.mdl.get_output_dims')
        dims['name'] = dims['name'].split(':')[-1]
        return dims

    def _get_current_output_dims(self, index: int):
        """Get the dimension of current output implementation.

        Args:
            index (int): The index of the output.
        """
        dims, ret = acl.mdl.get_cur_output_dims(self._desc, index)
        _check(ret, 'acl.mdl.get_cur_output_dims')
        return dims

    def get_current_ouptut_dims(self):
        """Get the dimension of current output."""
        dimses = []
        for output in self.outputs:
            dims = self._get_current_output_dims(output.index)
            dimses.append(dims['dims'])
        return dimses

    def _get_input_index(self, name: str) -> int:
        """Get input index by name.

        Args:
            name (str): The name of the input.

        Returns:
            (int): The input index.
        """
        index, ret = acl.mdl.get_input_index_by_name(self._desc, name)
        return index if ret == 0 else -1

    def get_dynamic_batch(self) -> Sequence:
        """Get dynamic batch size list.

        Returns:
            (Sequence): The dynamic batch list.
        """
        batch, ret = acl.mdl.get_dynamic_batch(self._desc)
        _check(ret, 'acl.mdl.get_dynamic_batch')
        batch = batch['batch']
        return sorted(batch)

    def get_dynamic_hw(self) -> Sequence:
        """Get dynamic height and width size list.

        Returns:
            (Sequence): The dynamic height and width
        """
        hw_info, ret = acl.mdl.get_dynamic_hw(self._desc, -1)
        _check(ret, 'acl.mdl.get_dynamic_hw')
        return hw_info['hw']

    def get_input_dynamic_dims(self) -> Sequence:
        """Get dynamic dims.

        Returns:
            (Sequence): The dynamic dims
        """
        count, ret = acl.mdl.get_input_dynamic_gear_count(self._desc, -1)
        _check(ret, 'acl.mdl.get_input_dynamic_gear_count')
        dims, ret = acl.mdl.get_input_dynamic_dims(self._desc, -1, count)
        _check(ret, 'acl.mdl.get_input_dynamic_dims')
        return dims


@BACKEND_WRAPPER.register_module(Backend.ASCEND.value)
class AscendWrapper(BaseWrapper):
    """Ascend wrapper class for inference.

    Args:
        model (str): Path of the model file.

    Examples:
        >>> from mmdeploy.backend.ascend import AscendWrapper
        >>> import torch
        >>>
        >>> model_file = 'model.om'
        >>> model = AscendWrapper(model_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
    """

    def __init__(self, model: str):

        self.context = AscendWrapper._init_context(0)

        self._model_id, ret = acl.mdl.load_from_file(model)
        _check(ret, 'acl.mdl.load_from_file')

        self._model_desc = ModelDesc(self._model_id)

        self._config_dynamic_shapes()
        self._create_input_buffers()
        self._create_output_buffers()

        output_names = [output.name for output in self._model_desc.outputs]

        super().__init__(output_names)

    def __del__(self):
        acl.mdl.unload(self._model_id)

    @classmethod
    def _init_context(cls, device_id):
        if hasattr(cls, 'context'):
            return cls.context

        _check(acl.init(), 'acl.init')
        _check(acl.rt.set_device(device_id), 'acl.rt.set_device')

        cls.context, ret = acl.rt.create_context(device_id)
        _check(ret, 'acl.rt.create_context')

        return cls.context

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """
        input_shapes = [inputs[x.name].shape for x in self._model_desc.inputs]

        output_shapes = self._reshape(input_shapes)

        for binding in self._model_desc.inputs:
            buffer_data = self._input.buffers[binding.index].data
            buffer_size = self._input.buffers[binding.index].size
            tensor = inputs[binding.name].contiguous().cpu().numpy()
            if tensor.dtype != binding.data_type:
                tensor = tensor.astype(binding.data_type)
            ptr, _ = tensor.__array_interface__['data']
            ret = acl.rt.memcpy(buffer_data, buffer_size, ptr, tensor.nbytes,
                                1)
            _check(ret, 'acl.rt.memcpy')

        self.__ascend_execute()

        outputs = {}
        for binding in self._model_desc.outputs:
            buffer_data = self._output.buffers[binding.index].data
            buffer_size = self._output.buffers[binding.index].size
            tensor = np.empty(
                output_shapes[binding.index], dtype=binding.data_type)
            ptr, _ = tensor.__array_interface__['data']
            ret = acl.rt.memcpy(ptr, tensor.nbytes, buffer_data, tensor.nbytes,
                                2)
            _check(ret, 'acl.rt.memcpy')
            outputs[binding.name] = torch.from_numpy(tensor)

        return outputs

    def _verify_dims(self, src: Sequence[int], ref: Sequence[int]):
        """Check if src match ref."""
        if len(src) != len(ref):
            raise RuntimeError(f'Shape mismatch {src} vs {ref}')
        for src_dim, ref_dim in zip(src, ref):
            if ref_dim != -1 and src_dim != ref_dim:
                raise RuntimeError(f'Shape mismatch {src} vs {ref}')

    def _reshape(self, input_shapes: Sequence[Sequence[int]]):
        """Reshape the inputs.

        Args:
            input_shapes (Sequence[Sequence[int]]): The shapes used to
                do reshape
        """

        if len(input_shapes) != len(self._model_desc.inputs):
            raise RuntimeError('#inputs mismatch')

        for src, ref in zip(input_shapes, self._model_desc.inputs):
            self._verify_dims(src, ref.dims)

        self._reshape_fn(input_shapes)

        dimses = self._model_desc.get_current_ouptut_dims()
        return dimses

    def _reshape_static(self, input_shapes):
        """Do nothing.

        Args:
            input_shapes (Sequence[Sequence[int]]): Not used.
        """
        pass

    def _reshape_dynamic_batch_size(self,
                                    input_shapes: Sequence[Sequence[int]]):
        """Reshape for dynamic batch size.

        Args:
            input_shapes (Sequence[Sequence[int]]): The shapes used to
                do reshape
        """
        batch_size = None
        for src, ref in zip(input_shapes, self._model_desc.inputs):
            if ref.dims[0] == -1:
                if batch_size is None:
                    batch_size = src[0]
                elif batch_size != src[0]:
                    raise RuntimeError(
                        f'Inconsistent batch size {batch_size} vs {src[0]}')

        if batch_size is None:
            raise RuntimeError('Can\'t determine batch size')

        candidates = list(
            filter(lambda x: x >= batch_size, self._dynamic_batch_size))
        if not candidates:
            raise RuntimeError(f'Batch size {batch_size} is not supported.'
                               f' ({self._dynamic_batch_size})')

        ret = acl.mdl.set_dynamic_batch_size(
            self._model_id, self._input.handle,
            self._model_desc.dynamic_tensor.index, candidates[0])
        _check(ret, 'acl.mdl.set_dynamic_batch_size')

    def _reshape_dynamic_image_size(self,
                                    input_shapes: Sequence[Sequence[int]]):
        """Reshape for dynamic image size.

        Args:
            input_shapes (Sequence[Sequence[int]]): The shapes used to
                do reshape
        """
        size = None
        for src, ref in zip(input_shapes, self._model_desc.inputs):
            if -1 in ref.dims:
                tmp_size = src[-2], src[-1]
                if size is None:
                    size = tmp_size
                elif size != tmp_size:
                    raise RuntimeError(
                        f'Inconsistent image size {size} vs {tmp_size}')

        if size is None:
            raise RuntimeError('Can\'t determine dynamic HW')
        if not list(size) in self._dynamic_hw:
            raise RuntimeError(
                f'size {size} is not supported. ({self._dynamic_hw})')
        height, width = size
        ret = acl.mdl.set_dynamic_hw_size(
            self._model_id, self._input.handle,
            self._model_desc.dynamic_tensor.index, height, width)
        _check(ret, 'acl.mdl.set_dynamic_hw_size')

    def _reshape_dynamic_dims(self, input_shapes: Sequence[Sequence[int]]):
        """Reshape for dynamic dims.

        Args:
            input_shapes (Sequence[Sequence[int]]): The shapes used to
                do reshape
        """
        match = [True] * len(self._dynamic_dims)
        ptr = 0
        for src in input_shapes:
            for axis, src_dim in enumerate(src):
                for index, dims in enumerate(self._dynamic_dims):
                    ref_dim = dims['dims'][ptr]
                    # allow batch dimension to vary
                    if axis == 0 and src_dim < ref_dim:
                        pass
                    elif src_dim != ref_dim:
                        match[index] = False
                ptr += 1

        indices = [i for i, v in enumerate(match) if v]
        if not indices:
            raise RuntimeError('No matching profile found')
        index = indices[0]

        ret = acl.mdl.set_input_dynamic_dims(
            self._model_id, self._input.handle,
            self._model_desc.dynamic_tensor.index, self._dynamic_dims[index])
        _check(ret, 'acl.mdl.set_input_dynamic_dims')

    def _config_dynamic_shapes(self):
        """Set the reshape function."""
        if self._model_desc.dynamic_tensor is None:
            self._reshape_fn = self._reshape_static
            return

        self._dynamic_batch_size = self._model_desc.get_dynamic_batch()
        if self._dynamic_batch_size:
            self._reshape_fn = self._reshape_dynamic_batch_size
            return

        self._dynamic_dims = self._model_desc.get_input_dynamic_dims()
        if self._dynamic_dims:
            self._reshape_fn = self._reshape_dynamic_dims
            return

        self._dynamic_hw = self._model_desc.get_dynamic_hw()
        if self._dynamic_hw:
            self._reshape_fn = self._reshape_dynamic_image_size
            return

        raise RuntimeError('Can\'t infer input shape type')

    def _create_input_buffers(self):
        """Create buffers for inputs."""
        self._input = Dataset()
        for binding in self._model_desc.inputs:
            self._input.add_buffer(DataBuffer(binding.size))
        if self._model_desc.dynamic_tensor:
            self._input.add_buffer(
                DataBuffer(self._model_desc.dynamic_tensor.size))

    def _create_output_buffers(self):
        """Create buffers for outputs."""
        self._output = Dataset()
        for binding in self._model_desc.outputs:
            self._output.add_buffer(DataBuffer(binding.size))

    @TimeCounter.count_time()
    def __ascend_execute(self):
        """Run inference on Ascend."""
        ret = acl.mdl.execute(self._model_id, self._input.handle,
                              self._output.handle)
        _check(ret, 'acl.mdl.execute')
