///*
// * Licensed to the Apache Software Foundation (ASF) under one or more
// * contributor license agreements.  See the NOTICE file distributed with
// * this work for additional information regarding copyright ownership.
// * The ASF licenses this file to You under the Apache License, Version 2.0
// * (the "License"); you may not use this file except in compliance with
// * the License.  You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package ml.dmlc.mxnetexamples.inferexample
//
//import ml.dmlc.mxnet._
//import ml.dmlc.mxnet.{DType, DataDesc}
//import ml.dmlc.mxnet.infer._
//// import ml.dmlc.mxnet.infer.Predictor
//import java.io.File
//import javax.imageio.ImageIO
//import java.awt.image.BufferedImage
//
//import org.slf4j.LoggerFactory
//
//import scala.io.Source
//import scala.collection.mutable.ListBuffer
//
//
//object PredictorExample {
//  private val logger = LoggerFactory.getLogger(classOf[PredictorExample])
//
//  def getScaledImage(img: BufferedImage): BufferedImage = {
//
//    val resizedImage = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB)
//    val g = resizedImage.createGraphics()
//    g.drawImage(img, 0, 0, 224, 224, null)
//
//    ImageIO.write(resizedImage, "jpg", new File("test.jpg"))
//    resizedImage
//  }
//
//  def getImage(imagePath: String): NDArray = {
//
//    val img = ImageIO.read(new File(imagePath))
//    val resizedImage = getScaledImage(img)
//
//    printf("Photo size is %d x %d\n", img.getWidth, img.getHeight)
//    printf("Photo size is %d x %d\n", resizedImage.getWidth, resizedImage.getHeight)
//
//    val w = resizedImage.getWidth
//    val h = resizedImage.getHeight
//
//    // create new image of the same size
//    val test = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
//
//    val pixels = new ListBuffer[Float]()
//
//    for (x <- 0 until h) {
//      for (y <- 0 until w) {
//        val color = resizedImage.getRGB(y, x)
//        val red = (color & 0xff0000) >> 16
//        val green = (color & 0xff00) >> 8
//        val blue = color & 0xff
//        pixels += red
//        pixels += green
//        pixels += blue
//      }
//    }
//
//    val reshaped_pixels = NDArray.array(pixels.toArray, shape = Shape(224, 224, 3))
//
//    val swapped_axis = NDArray.swapaxes(reshaped_pixels, 0, 2)
//    val y = NDArray.swapaxes(swapped_axis, 1, 2)
//
//    y
//  }
//
//  def createLabels(filename: String): List[String] = {
//      val lines = Source.fromFile(filename).getLines.toList
//      lines
//  }
//
//  def main(args: Array[String]): Unit = {
//
//    val dType = DType.Float32
//    val inputShape = Shape(1, 3, 224, 224)
//    val outputShape = Shape(1, 1000)
//
//    val modelPathPrefix = "/Users/roshanin/Downloads/resnet/resnet-152"
//    val filePath = "/Users/roshanin/Downloads/Cat-hd-wallpapers.jpg"
//    val synset = "/Users/roshanin/Downloads/resnet/synset.txt"
//
//    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NT"))
//    val outDescriptor = Option(IndexedSeq(DataDesc("data", outputShape, dType, "NT")))
//
//    printf("%s", inputDescriptor(0).shape(2))
//
//    val predictor: PredictBase = new Predictor(modelPathPrefix, inputDescriptor, outDescriptor)
//
//    val pixels = getImage(filePath)
//
//    val input = IndexedSeq(pixels.reshape(Shape(1, 3, 224, 224)))
//
//    val predictResult = predictor.predictWithNDArray(input)
//
//    val prob = predictResult(0)
//
//    // create labels array from synset
//    val labels = createLabels(synset)
//
//    // get predicted labels
//    val topProbs = NDArray.argsort(prob).toArray.reverse.slice(0, 5)
//
//    val probArray = prob.toArray
//
//    for (i <- topProbs) {
//      printf("probability= %f, class=%s \n", probArray(i.asInstanceOf[Int]),
//        labels(i.asInstanceOf[Int]))
//    }
//  }
//}
//
//class PredictorExample {
////  @Option(name = "--data-dir", usage = "the input data directory")
////  private val dataDir: String = "mnist/"
//
//}
