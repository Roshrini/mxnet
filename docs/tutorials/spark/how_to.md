# How to start using MXNet on Spark

## Introduction

### Why MXNet

Deep learning and neural networks are increasingly important concepts in many areas of our businesses and our lives

Machine learning is playing an increasingly important role in many areas of our businesses and our lives and is being employed in a range of computing tasks where programming explicit algorithms is infeasible. Among machine learning algorithms, a class of algorithms called Deep Learning and Neural Networks is the hottest area in the market right now. But training deep neural nets can take precious time and resources. It's computationally very expensive.

Among the multiple deep learning frameworks available out there, MXNet is the most scalable framework. It provides both the flexible parallel training approaches and GPU support for faster computation. So, why use Spark?

###  Why [Apache Spark](http://spark.apache.org/)

Apache Spark is the distributed big data processing framework built around speed, ease of use, and sophisticated analytics. Spark gives us a comprehensive, unified framework to manage big data processing requirements with a variety of data sets that are diverse in nature (text data, graph data etc) as well as the source of data (batch v. real-time streaming data). Spark enables applications in Hadoop clusters to run up to 100 times faster in memory and 10 times faster even when running on disk.

### Use-cases for MXNet on Spark

There can be two use cases to explain how you can use Spark and a cluster of machines to increase efficiency of deep learning algorithms with MXNet.

* Data processing in Spark and distributed training with MXNet : We can leverage Spark for its lightening fast execution engine and in-memory computing to do big data processing. Program will take input as RDDs and then, feed the data to do distributed training using MXNet.

* Running pre-trained models: Use Spark to apply a pre-trained neural network model on a large amount of data for inference.

In this way, Users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark cluster. By leveraging an existing distributed batch processing framework, MXNet can train neural nets quickly and efficiently.

Now, you have an end-to-end solution for large-scale deep models, which means you can take advantage of both frameworks to build the full pipeline from dealing with the raw data to build the efficient deep learning algorithms.

## MXNet Spark package

Currently, It is built on the MXNet Scala Package and brings deep learning to Spark. The MXNet on Spark is still in experimental stage. Any suggestion or contribution will be highly appreciated.

### How to install

Follow the [setup and installation for MXNet](http://mxnet.io/get_started/index.html#setup-and-installation).
Remember to enable the distributed training, i.e., set USE_DIST_KVSTORE = 1.
Install MXNet package for Scala like [this](http://mxnet.io/get_started/ubuntu_setup.html#install-the-mxnet-package-for-scala).

When you build the MXNet Scala package with `make` command, Spark package automatically gets installed in your machine. You can find the package in `mxnet/scala-package/spark` folder. MXNet Spark integration jar is present in the `mxnet/scala-package/spark/target` folder. You will need to include this jar while running your application.

It's very easy to install Apache Spark on cluster of machines or on single node.

* You can download Spark pre-build version or source code from [Apache Spark website](http://spark.apache.org/downloads.html).
* If you download source code, you will need to [build it from source](http://spark.apache.org/docs/latest/building-spark.html). Default Scala version is 2.11 for Spark 2.x. For some reason if you want to use Scala version 2.10, you can do that while running Spark directory from source.

```bash
    ./dev/change-scala-version.sh 2.10
    ./build/mvn -Pyarn -Phadoop-2.4 -Dscala-2.10 -DskipTests clean package
```

* Do not forget to set `SPARK_HOME` environment variable. It should point to the downloaded/built source directory of Apache Spark.


### Running Spark in Cluster mode

Apache Spark supports three cluster managers. You can find the details about  [Cluster Managers here](http://spark.apache.org/docs/latest/cluster-overview.html#cluster-manager-types). Currently MXNet Spark package supports only cluster mode. So, you must run your Spark job in cluster mode (standalone, yarn-client, yarn-cluster). Follow the links to setup your Spark cluster.

* Follow this [to setup Spark Standalone cluster](http://spark.apache.org/docs/latest/spark-standalone.html).

(Hint: If you only have one physical node and want to test the Spark package, you can start N workers on one node by setting export SPARK_WORKER_INSTANCES=N in spark-env.sh.)

After you get the running cluster, You can find `<master-spark-url>` from the Spark UI, mention the same while submitting the job through spark-submit.

* Follow this [to use Yarn](http://spark.apache.org/docs/latest/running-on-yarn.html) cluster manager for your cluster.

## How to run the existing example

You can find the example showing how to apply Multi-Layer Perceptron and Lenet algorithm on MNIST dataset in Spark and do training in MXNet.
You can submit the jobs to Spark standalone cluster through spark-submit. Here is how to do it:

```bash
    {SPARK_HOME}/bin/spark-submit --master <master-spark-url> \
      --conf spark.dynamicAllocation.enabled=false \
      --conf spark.speculation=false \
      --class <Example to run> \
      <Spark configuration options> \
      <MXNet Spark jar created > \
      <Command-line options required for application>
```

e.g.,  

```bash
    ${SPARK_HOME}/bin/spark-submit --master spark://localhost:7077 \
      --conf spark.dynamicAllocation.enabled=false \
      --conf spark.speculation=false \
      --class ml.dmlc.mxnet.spark.example.ClassificationExample \
      --name mxnet --driver-memory 1g --executor-memory 1g --num-executors 2 --executor-cores 1 \
      --jars /path-to-mxnet-jar,spark-jar \
      /target/mxnet-spark_2.11-0.1.2-SNAPSHOT.jar \
      ${RUN_OPTS}
```

To run the mnist example, first make changes to the [script](https://github.com/dmlc/mxnet/tree/master/scala-package/spark/bin) as follows:
* Make sure `SPARK_HOME` variable is set.
* Change path to `mxnet-full_2.10-linux-x86_64-cpu-0.1.2-SNAPSHOT.jar` jar according to your MXNet configuration.
* Change JAR, LIBS variables according to the Scala version on your machine. Make sure to mention correct names of jars `args4j-2.0.29.jar`, `scala-library-2.10.4.jar` and `mxnet-spark_2.10-0.1.2-SNAPSHOT.jar`.
* Set java path here 'RUN_OPTS'
* Set INPUT_TRAIN(input training dataset path) , INPUT_VAL(input validation dataset path), OUTPUT(name of directory to save output) variables in the script.

Then, you can directly run the example as follows:

```bash
    ./run-mnist-example.sh
```

### Explanation

#### Spark context creation

Create a spark context variable in Spark

```scala
    val conf = new SparkConf().setAppName("MXNet")
    val sc = new SparkContext(conf)
```

#### Data Processing

Now, you can load data to do preprocessing using spark functions (`map`, `reduce`, `group by` etc. )

```scala
    val trainData = parseRawData(sc, cmdLine.input)
    val valData = parseRawData(sc, cmdLine.inputVal)
```

#### Distributed Training

Training can be done by creating object of MXNet class and setting the required parameters as follows:

```scala
    val mxnet = new MXNet()
            .setBatchSize(128)    // batch size for number of examples
            .setLabelName("softmax_label")
            .setContext(Context.cpu()) //number of cpus/gpus to use for training
            .setDimension(dimension)   //dimension needed for model
            .setNetwork(network)       // model e.g MLP/Lenet
            .setNumEpoch(10) // training iterations
            .setNumServer(2)
            .setNumWorker(4)
            .setExecutorJars(cmdLine.jars)
            .setJava(cmdLine.java)
```

`setExecutorJars` function will take care of uploading and distributing those jars (`args4j-2.0.29.jar`, `scala-library-2.11.8.jar`, `mxnet-spark_2.11-0.1.2-SNAPSHOT.jar`) to each node automatically. These jars are required by the KVStores at runtime.

Here, Parameter Server (PS) scheduler will start on driver and by specifying number of servers as 2 means Parameter Server (PS) will launch 2 servers. This is done internally by calling method `sc.parallelize(1 to numServer, numServer)`. So, stage will have 2 tasks while running the job.

Similarly, by specifying number of workers as 4, the input data will be split into 4 pieces. This is done internally by using spark's `coalesce` and `repartition` operators. If the RDD had less partitions than number of workers mentioned, data is repartitioned and if the RDD had more partitions than number of workers mentioned, data is coalesced. This is important because a stage in Spark will operate on one partition at a time (and load the data in that partition into memory).

Now, You can do distributed training as follows:

```scala
    val model = mxnet.fit(trainData)
```

MXNet uses PS-lite package for the implementation of [Parameter Server](https://github.com/dmlc/ps-lite). PS-Lite provides asynchronous communication for MXNet. MXNet on spark is using [dist_async mode](http://ps-lite.readthedocs.io/en/latest/overview.html#asynchronous-sgd) in PS-lite. So, it takes care of the cases where failed nodes in Spark are restarted automatically and lost workers are reconnected to PS scheduler and servers.


#### Save the model

To save the output model, simply call save method and pass spark context variable and output folder path.

```scala
    model.save(sc, cmdLine.output + "/model")
```

#### Distribute the model to all the worker nodes:

This can be done by using [Broadcast variables](http://spark.apache.org/docs/latest/programming-guide.html#broadcast-variables) in Spark.
Broadcast variables are immutable and can be distributed to the cluster of machines.

They can be created from a variable v by calling `sc.broadcast(v)`. The broadcast variable is a wrapper around v, and its value can be accessed by calling the `value` method. The code below shows this:

```scala
    val brModel = sc.broadcast(model)
```

So, here model is an immutable (read-only) variable cached on each worker machine rather than shipping a copy of it with tasks. Its useful because model will be distributed once per node using efficient p2p protocol and you'll get huge performance benefit.

#### Prediction on validation dataset

Predicting is straightforward, You can use simple Scala function with Spark to do this. `predict` function will predict the probability of the label for all the examples given to it as input.

```scala
    val res = valData.mapPartitions { data =>
      val probArrays = brModel.value.predict(points.toIterator)
      require(probArrays.length == 1)
      val prob = probArrays(0)
      val py = NDArray.argmaxChannel(prob.get)
      val labels = py.toArray.mkString(",")
      py.dispose()
      prob.get.dispose()
      labels
    }
```

Output can be saved using Spark method `saveAsTextFile` as follows:

```scala
    res.saveAsTextFile(cmdLine.output + "/data")
```

#### Load pre-trained model

You can also load the existing pre-trained model in your job as follows:

```scala
    val model = MXNetModel.load(sc, cmdLine.inputModel)
```

 You will need to broadcast this model to all the workers and then draw inference from it as explained above.

## Future Work

* MXNet and ps-lite currently do NOT support multiple instances in one process. Local mode is NOT supported because it runs tasks in multiples threads with one process, which will block the initialization of KVStore.

## Next Steps
* [Spark Guide](https://github.com/dmlc/mxnet/tree/master/scala-package/spark)
* [Scala API](http://mxnet.io/api/scala/)
* [More Scala Examples](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples)
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
