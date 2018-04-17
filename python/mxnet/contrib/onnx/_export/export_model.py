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
"""export function"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import defs, checker, helper, numpy_helper, mapping
from .export_onnx import MxNetToONNXConverter
from .export_helper import load_module

import numpy as np

def export_model(model, weights, input_shape, input_type, log=False):
    """Exports the MXNet model file, passed as a parameter, into ONNX model.
    Accepts both symbol,parameter objects as well as json and params filepaths as input.
    Operator support and coverage - https://cwiki.apache.org/confluence/display/MXNET/ONNX

    Parameters
    ----------
    model : str or symbol object
        Path to the json file or Symbol object
    weights : str or symbol object
        Path to the params file or Params object. (Including both arg_params and aux_params)
    input_shape :
        Input shape of the model e.g (1,3,224,224)
    input_type :
        Input data type e.g. np.float32
    log : Boolean
        If true will print logs of the model conversion

    Returns
    -------
    onnx_model : onnx ModelProto
        Onnx modelproto object
    """
    converter = MxNetToONNXConverter()

    if isinstance(model, basestring) and isinstance(weights, basestring):
        print("Converting json and params file to sym and weights")
        sym, params = load_module(model, weights, input_shape)
        onnx_graph = converter.convert_mx2onnx_graph(sym, params, input_shape,
                                                     mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(input_type)], log=log)
    else:
        onnx_graph = converter.convert_mx2onnx_graph(model, weights, input_shape,
                                                 mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(input_type)], log=log)
    # Create the model (ModelProto)
    onnx_model = helper.make_model(onnx_graph)
    return onnx_model
