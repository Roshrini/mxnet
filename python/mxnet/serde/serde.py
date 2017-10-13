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
from .import_onnx import GraphProto

def import_from(graph, format):
    """Load onnx graph which is a python protobuf object in to nnvm graph.
    The companion parameters will be handled automatically.
    The inputs from onnx graph is vague, only providing "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    Parameters
    ----------
    graph : protobuf object
        ONNX graph

    Returns
    -------
    sym : mx.symbol
        Compatible mxnet symbol

    params : dict of str to mx.ndarray
        Dict of converted parameters stored in mx.ndarray format
    """
    if format == 'onnx':
        g = GraphProto()
        sym, params = g.from_onnx(graph)
        return sym, params
    else:
        raise NotImplementedError("Only onnx format is supported")

def export_to(graph, format):
    pass