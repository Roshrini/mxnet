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

from __future__ import absolute_import as _abs
import os
import mxnet as mx
import onnx
import numpy as np

from mxnet.serde import serde

def _as_abs_path(fname):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cur_dir, fname)

def download(url, path, overwrite=False):
    import urllib2, os
    if os.path.exists(path) and not overwrite:
        return
    print('Downloading {} to {}.'.format(url, path))
    with open(path, 'w') as f:
        f.write(urllib2.urlopen(url).read())

onnx_graph = onnx.load(_as_abs_path('super_resolution.onnx'))
# sym, params = import_onnx.from_onnx(onnx_graph)
# print sym.list_arguments()

sym, params = serde.import_from(onnx_graph, 'onnx')
print sym

#Load test image
# from PIL import Image
# img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
# download(img_url, 'cat.png')
# img = Image.open('cat.png').resize((224, 224))
# img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
# img_y, img_cb, img_cr = img_ycbcr.split()
# x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

# create module
# mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
# mod.bind(for_training=False, data_shapes=[('input_0',x.shape)], label_shapes=mod._label_shapes)
# mod.set_params(arg_params=params, allow_missing=True)