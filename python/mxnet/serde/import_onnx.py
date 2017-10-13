# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
""" Support import export formats."""
from __future__ import absolute_import as _abs
from .common import Renamer, AttrConverter as AttrCvt
from .. import ndarray as _nd
from .. import symbol as _sym

def _revert_caffe2_pad(attr):
    """Caffe2 require two times the normal padding."""
    if len(attr) == 4:
        attr = attr[:2]
    elif len(attr) == 2:
        pass
    else:
        raise ValueError("Invalid caffe2 type padding: {}".format(attr))
    return attr

def _math_name_picker(surfix):
    def _impl(attr):
        if attr.get('broadcast', 0):
            return 'broadcast_' + surfix
        return 'elemwise_' + surfix
    return _impl

def _broadcast_constraint():
    def _broadcast_check(attrs):
        if attrs.get('axis', None):
            return False
        return True
    return _broadcast_check, "Specifying broadcast axis not allowed."

def _dimension_picker(prefix, surfix=''):
    def _impl(attr):
        kernel = attr['kernel_shape']
        # if len(kernel) == 2:
        #     return len(kernel)+''
        # else:
        #     raise NotImplementedError("Only 2d kernel supported.")
        return len(kernel)
    return _impl

def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) == 2:
            return True
        return False
    return _dim_check, "Only 2d kernel supported."

def _elemwise(name):
    return AttrCvt(
        op_name=_math_name_picker(name),
        disables=['axis'],
        ignores=['broadcast'])

# pooling types need to be checked [TODO]
def _pooling(name):
    return AttrCvt(
        op_name='Pooling',
        transforms={
            'kernel_shape': 'kernel',
            'pool_type': name,
            'strides': 'stride',
            'pads': ('pad', (0, 0), _revert_caffe2_pad)},
        # very weird attributes here in onnx, force check
        ignores=['dilations'],
        custom_check=_dimension_constraint())

# Requires kernel attribute which is not present in onnx
def _global_pooling(name):
    return AttrCvt(
        op_name='Pooling',
        transforms={
            'kernel_shape': 'kernel',
            'pool_type': name},
    extras={'global_pool': True},
    custom_check = _dimension_constraint())

def _conv():
  #  dim = _dimension_picker('conv')
   # print int(dim)
    return AttrCvt(
        op_name='Convolution',
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'dilations': ('dilate', (0,0)),
            'pads': ('pad', (0,0), _revert_caffe2_pad),
            'group': ('num_group', 1)},
        custom_check=_dimension_constraint())

def _conv_transpose():
    return AttrCvt(
        op_name='Deconvolution',
        #_dimension_picker('conv', '_transpose'),
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'dilations': ('dilate', (0, 0)),
            'pads': ('pad', (0, 0), _revert_caffe2_pad)},
        disables=['output_shape'],
        custom_check=_dimension_constraint())

def _batch_norm():
    # TODO(zhreshold): 'spatial' is not properly handled here.
    return AttrCvt(
        op_name='BatchNorm',
        transforms={'epsilon':'eps'},
        ignores=['spatial', 'is_test'])

def _arg_op(name):
    return AttrCvt(
        op_name=name,
        transforms={
            'axes':'axis'})

def _activation(name):
    return AttrCvt(
        op_name='LeakyReLU',
        transforms={
            'alpha':'slope'},
        extras={'act_type': name})

# Only 4d 5d tensor supported in mxnet. check[TODO]
def _pad():
    return AttrCvt(
        op_name='pad',
        transforms={
            'paddings':'pad_width',
            'value':'constant_value'})

# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
_convert_map = {
    # defs/experimental
    'FC'            : AttrCvt('FullyConnected', ignores=['axis', 'axis_w']),

    # defs/generator
    # 'Constant'
    'RandomUniform' : AttrCvt('random_uniform', ignores=['seed']),
    'RandomNormal'  : AttrCvt('random_normal', {'mean':'loc'}, ignores=['seed']),
    'RandomUniformLike' : AttrCvt('random_uniform', ignores=['seed']),
    'RandomNormalLike': AttrCvt('random_normal', {'mean':'loc'}, ignores=['seed']),

    # defs/logical

    # defs/math
    'Add'           : _elemwise('add'),
    'Sub'           : _elemwise('sub'),
    'Mul'           : _elemwise('mul'),
    'Div'           : _elemwise('div'),
    'Neg'           : Renamer('negative'),
    'Abs'           : Renamer('abs'),
    'Reciprocal'    : Renamer('reciprocal'),
    'Floor'         : Renamer('floor'),
    'Ceil'          : Renamer('ceil'),
    'Sqrt'          : Renamer('sqrt'),
    'Gemm'          : AttrCvt('linalg_gemm', {'transA':'transpose_a', 'transB':'transpose_b'}, ignores=['broadcast']),
    'Relu'          : Renamer('relu'),
    'LeakyRelu'     : Renamer('leaky'),
    # 'Selu'
    'Elu'           : _activation('elu'),
    'Exp'           : Renamer('exp'),
    'Log'           : Renamer('log'),
    'Tanh'          : Renamer('tanh'),
    'Pow'           : AttrCvt('pow', {'exponent':'exp'}),
    'Dot'           : Renamer('dot'),
    # 'PRelu'
    'Sigmoid'       : Renamer('sigmoid'),
    'Max'           : Renamer('maximum'), #elemwise maximum
    'Min'           : Renamer('minimum'), #elemwise minimum
    'Sum'           : Renamer('add_n'), #elemwise sum
    # softmax default axis is different in onnx
    'Softmax'       : AttrCvt('softmax', {'axis': ('axis', 1)}),

    # defs/nn
    'AveragePool'   : _pooling('avg'),
    'MaxPool'       : _pooling('max'),
    'Conv'          : _conv(),
    'ConvTranspose' : _conv_transpose(),
    # 'GlobalAveragePool': _global_pooling('avg'),
    # 'GlobalMaxPool' : _global_pooling('max'),
    'BatchNormalization': _batch_norm(),
    'Dropout'       : AttrCvt('dropout', {'ratio': 'p'}, ignores=['is_test']),
    'Flatten'       : Renamer('flatten'),

    # defs/reduction
    'ReduceMax'     : AttrCvt('max', {'axes', 'axis'}),
    'ReduceMin'     : AttrCvt('min', {'axes', 'axis'}),
    'ReduceSum'     : AttrCvt('sum', {'axes', 'axis'}),
    'ReduceMean'    : AttrCvt('mean', {'axes', 'axis'}),
    'ReduceProd'    : AttrCvt('prod', {'axes', 'axis'}),
    # 'ReduceLogSumExp'
    # 'ArgMax'        : _arg_op('argmax'),
    # 'ArgMin'        : _arg_op('argmin'),

    # defs/tensor
    'Cast'          : AttrCvt('cast', {'to': 'dtype'}),
    'Reshape'       : Renamer('reshape'),
    'Concat'        : AttrCvt('concat', {'axis': 'dim'}),
    'Split'         : AttrCvt('split', {'split': 'num_outputs'}),
    'Pad'           : _pad(),
    'Slice'         : AttrCvt('slice', {'ends': 'end', 'starts': 'begin'}),
    'Transpose'     : AttrCvt('transpose', {'perm': 'axes'}),
    # 'Gather'
    # 'Squeeze'
}

def _convert_operator(op_name, attrs, identity_list=None, convert_map=None):
    """Convert from onnx operator to mxnet operator.
    The converter must specify conversions explicity for incompatible name, and
    apply handlers to operator attributes.

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, FullyConnected
    attrs : dict
        Dict of operator attributes
    identity_list : list
        List of operators that don't require conversion
    convert_map : dict
        Dict of name : callable, where name is the op's name that
        require conversion to mxnet, callable are functions which
        take attrs and return (new_op_name, new_attrs)

    Returns
    -------
    (op_name, attrs)
        Converted (op_name, attrs) for mxnet.
    """
    identity_list = identity_list if identity_list else _identity_list
    convert_map = convert_map if convert_map else _convert_map
    if op_name in identity_list:
        pass
    elif op_name in convert_map:
        op_name, attrs = convert_map[op_name](attrs)
    else:
        _raise_not_supported('Operator: ' + op_name)
    op = getattr(_sym, op_name, None)
    if not op:
        raise RuntimeError("Unable to map op_name {} to sym".format(op_name))
    return op, attrs


class GraphProto(object):
    """A helper class for handling mxnet symbol copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0

    def from_onnx(self, graph):
        """Construct symbol from onnx graph.
        The inputs from onnx graph is vague, only providing "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        Returns
        -------
        sym :mx.symbol
            The returned mxnet symbol
        params : dict
            A dict of name: mx.nd.array pairs, used as pretrained weights
        """
        # parse network inputs, aka parameters
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self._params[init_tensor.name] = self._parse_array(init_tensor)
        # converting GraphProto message
        for i in graph.input:
            if i in self._params:
                # i is a param instead of input
                name_param = 'param_{}'.format(self._num_param)
                self._num_param += 1
                self._params[name_param] = self._params.pop(i)
                self._nodes[name_param] = _sym.Variable(name=name_param)
                self._renames[i] = name_param
            else:
                name_input = 'input_{}'.format(self._num_input)
                self._num_input += 1
                self._nodes[name_input] = _sym.Variable(name=name_input)
                self._renames[i] = name_input
        # construct nodes, nodes are stored as directed acyclic graph
        # converting NodeProto message
        for idx, node in enumerate(graph.node):
            op_name = node.op_type
            node_name = node.name.strip()
            node_name = node_name if node_name else None
            attr = self._parse_attr(node.attribute)
            new_op, new_attr = _convert_operator(op_name, attr)
            inputs = [self._nodes[self._renames.get(i, i)] for i in node.input]
            # some hacks for onnx problem
            new_attr = self._fix_bias(new_op, new_attr, len(inputs))
            new_attr = self._fix_channels(new_op, new_attr, list(node.input))
            self._fix_bias_shape(node.op_type, graph.node[idx - 1].op_type, node.input)
            op = new_op(name=node_name, *inputs, **new_attr)
            node_output = self._fix_outputs(op_name, node.output)
            assert len(node.output) == len(op.list_outputs()), (
                "Number of output mismatch {} vs {} in {}.".format(
                    len(node_output), len(op.list_output_names()),op_name))
            for k, i in zip(list(node_output), range(len(node_output))):
                self._nodes[k] = op[i]
                # now return the outputs
        out = [self._nodes[i] for i in graph.output]
        if len(out) > 1:
            out = _sym.Group(out)
        else:
            out = out[0]
        return out, self._params

    def _parse_array(self, tensor_proto):
        """Grab data in TensorProto and convert to numpy array."""
        try:
            from onnx.numpy_helper import to_array
        except ImportError as e:
            raise ImportError("Unable to import onnx which is required {}".format(e))
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return _nd.array(np_array)

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ['f', 'i', 's']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ['floats', 'ints', 'strings']:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ['t', 'g']:
                if a.HasField(f):
                    raise NotImplementedError("Filed {} is not supported in mxnet.".format(f))
            for f in ['tensors', 'graphs']:
                if list(getattr(a, f)):
                    raise NotImplementedError("Filed {} is not supported in mxnet.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _fix_outputs(self, op, outputs):
        """A hack to handle dropout or similar operator that have more than one out
        in ONNX.
        """
        if op == 'Dropout':
            assert len(outputs) == 2, "ONNX have two outputs for dropout layer."
            outputs = outputs[:-1]
        return outputs

    def _fix_bias(self, op, attrs, num_inputs):
        """A hack for 'use_bias' attribute since onnx don't provide this attribute,
        we have to check the number of inputs to decide it."""
        if op not in [_sym.Convolution, _sym.Deconvolution, _sym.FullyConnected]:
            return attrs
        if num_inputs == 3:
            attrs['bias'] = True
        elif num_inputs == 2:
            attrs['no_bias'] = True
        else:
            raise ValueError("Unexpected number of inputs for: {}".format(op))
        return attrs


    def _fix_bias_shape(self, op_name, last_op_name, inputs):
        """A hack to reshape bias term to (1, num_channel)."""
        if op_name == 'Add' and last_op_name == 'Conv':
            assert len(list(inputs)) == 2
            bias_name = self._renames.get(inputs[1], inputs[1])
            bias = self._params[bias_name]
            assert len(bias.shape) == 1
            # reshape to (1, n)
            bias = _nd.array(bias.asnumpy().reshape((1, -1, 1, 1)))
            self._params[bias_name] = bias


    def _fix_channels(self, op, attrs, inputs):
        """A hack for getting 'channels' or 'units' since onnx don't provide
        these attributes. We check the shape of weights provided to get the number.
        """
        if op not in [_sym.Convolution, _sym.Deconvolution, _sym.FullyConnected]:
            return attrs
        weight_name = self._renames[inputs[1]]
        if not weight_name in self._params:
            raise ValueError("Unable to get channels/units attr from onnx graph.")
        else:
            wshape = self._params[weight_name].shape
            assert len(wshape) >= 2, "Weights shape is invalid: {}".format(wshape)
            channels = wshape[0]
            if op in [_sym.FullyConnected]:
                attrs['num_hidden'] = channels
            else:
                attrs['num_filter'] = channels
        return attrs