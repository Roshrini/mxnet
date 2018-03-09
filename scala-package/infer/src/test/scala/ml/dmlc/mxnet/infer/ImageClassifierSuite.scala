/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnet.infer

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.module.Module
import ml.dmlc.mxnet.io.NDArrayIter
import ml.dmlc.mxnet.{DType, DataDesc, Shape}
import org.mockito.Matchers._
import org.mockito.Mockito
import org.scalatest.{BeforeAndAfterAll, FunSuite, Ignore}
import java.awt.Image
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.{Files, Paths}
import java.util

class ImageClassifierSuite extends ClassifierSuite with BeforeAndAfterAll {

  class MyImageClassifier(modelPathPrefix: String,
                           inputDescriptors: IndexedSeq[DataDesc])
    extends ImageClassifier(modelPathPrefix, inputDescriptors) {

    override def getPredictor(modelPathPrefix: String,
      inputDescriptors: IndexedSeq[DataDesc]): PredictBase = {
      Mockito.mock(classOf[MyClassyPredictor])
    }

    override def getClassifier(modelPathPrefix: String, inputDescriptors: IndexedSeq[DataDesc]):
    Classifier = {
      Mockito.mock(classOf[Classifier])
    }
  }

  test("Rescale an image") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(2, 3, 2, 2)))
    val testImageClassifier =
        new MyImageClassifier(modelPath, inputDescriptor)

    val image1 = new BufferedImage(100, 200, BufferedImage.TYPE_BYTE_GRAY)
    val image2 = testImageClassifier.getScaledImage(image1, 1000, 2000)

    assert(image2.getWidth === 1000)
    assert(image2.getHeight === 2000)
  }

  test("Convert BufferedImage to NDArray") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 2, 2)))
    val testImageClassifier =
      new MyImageClassifier(modelPath, inputDescriptor)

    val image1 = new BufferedImage(100, 200, BufferedImage.TYPE_BYTE_GRAY)
    val image2 = testImageClassifier.getScaledImage(image1, 224, 224)

    val result = testImageClassifier.getPixelsFromImage(image2)

   // assert(result.getClass === NDArray.getClass)
  }

  test("testWithInputImage") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 224, 224)))

    val inputImage = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB)

    val predictResult : IndexedSeq[Array[Float]] =
      IndexedSeq[Array[Float]](Array(.98f, 0.97f, 0.96f, 0.99f))

    val testImageClassifier: ImageClassifier =
      new MyImageClassifier(modelPath, inputDescriptor)

  //  val spy = Mockito.spy(testImageClassifier)

    Mockito.doReturn(predictResult).when(testImageClassifier.classifier)
      .classifyWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    val result = testImageClassifier.classifyImage(inputImage)

    //    assertResult(predictResult(0)) {
//      result.map(_._2).toArray
//    }
  }

  test("testWithInputBatchImage") {

  }
}
