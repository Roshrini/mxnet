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

"""ONNX test backend wrapper"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
try:
    import onnx.backend.test
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed")

import backend as mxnet_backend

# This is a pytest magic variable to load extra plugins
pytest_plugins = "onnx.backend.test.report",

BACKEND_TEST = onnx.backend.test.BackendTest(mxnet_backend, __name__)

IMPLEMENTED_OPERATORS_TEST = [
    #Arithmetic Operators
    'test_add'
    'test_sub',
    'test_mul',
    'test_div',
    'test_neg',
    'test_abs',
    'test_sum',

    # Hyperbolic functions
    'test_tanh',

    #Rounding
    'test_ceil',
    'test_floor',

    # Basic neural network functions
    'test_sigmoid',
    'test_relu',
    # 'test_elu',
    'test_constant_pad',
    'test_edge_pad',
    'test_reflect_pad',
    'test_conv',
    'test_basic_conv',
    ]

BASIC_MODEL_TESTS = [
    # 'test_AvgPool2D',
    'test_BatchNorm',
    # 'test_ConstantPad2d'
    # 'test_Conv2d',
    # 'test_ELU',
    # 'test_LeakyReLU',
    # 'test_MaxPool',
    # 'test_PReLU',
    'test_ReLU',
    'test_Sigmoid',
    # 'test_Softmax',
    # 'test_softmax_functional',
    # 'test_softmax_lastdim',
    'test_Tanh'
    ]

STANDARD_MODEL = [
    'test_bvlc_alexnet',
    'test_densenet121',
    #'test_inception_v1',
    #'test_inception_v2',
    'test_resnet50',
    #'test_shufflenet',
    'test_squeezenet',
    'test_vgg16',
    'test_vgg19'
    ]

for op_test in IMPLEMENTED_OPERATORS_TEST:
    BACKEND_TEST.include(op_test)

for basic_model_test in BASIC_MODEL_TESTS:
    BACKEND_TEST.include(basic_model_test)

# for std_model_test in STANDARD_MODEL:
#     BACKEND_TEST.include(std_model_test)


# import all test cases at global scope to make them visible to python.unittest
globals().update(BACKEND_TEST.enable_report().test_cases)

if __name__ == '__main__':
    unittest.main()