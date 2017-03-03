# Run MXNet Scala examples using IDE

This Scala tutorial guides you through setting up Scala project in IntelliJ IDE and shows how to use MXNet package from your application.

## Prerequisites:

- [Maven 3](https://maven.apache.org/install.html)
- [Scala 2.11.8](https://www.scala-lang.org/download/2.11.8.html)
- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/get_started/setup.html#overview)
- Install MXNet package for Scala. Follow steps [here](http://mxnet.io/get_started/osx_setup.html#install-the-mxnet-package-for-scala)
- [IntelliJ IDE](https://www.jetbrains.com/idea/)

## Set up your project

- Install Scala plugin for IntelliJ IDE. You can do this by following these simple steps.
 Go to Menu -> Preferences -> Select Plugins -> Type Scala -> Click on install

- Follow [Scala plugin setup for IDE](https://www.jetbrains.com/help/idea/2016.3/scala.html) for more details.

- When you build your MXNet package with Scala, MXNet-Scala jar is generated in `native/{your-architecture}/target` directory which you will need to use while creating an example package having MXNet as dependency.

- Specify dependencies needed for your project in pom.xml. To use MXNet Scala package, add following dependencies

```bash
    <dependencies>
      <dependency>
        <groupId>ml.dmlc.mxnet</groupId>
        <artifactId>mxnet-full_${scala.binary.version}-${platform}</artifactId>
        <version>0.1.1</version>
        <scope>system</scope>
        <systemPath>`MXNet-Scala-jar-path`</systemPath>
      </dependency>
      <dependency>
        <groupId>args4j</groupId>
        <artifactId>args4j</artifactId>
        <version>2.0.29</version>
      </dependency>
    </dependencies>
```

Make sure to change system path of MXNet-Scala jar which is in `native/{your-architecture}/target` directory.

- Right click on your example project, click on Maven and then reimport. These steps will add all the dependencies you mentioned in pom.xml as external libraries in your project.

- To build the project, Go to Menu, click on Build -> Rebuild Project. Solve any errors shown in IDE.

- You can also compile the project with the following command from command line.

```bash
    cd mxnet-scala-example
    mvn clean package
```

- This will also generate mxnet-scala-example-0.1-SNAPSHOT.jar file for your application.

Check out more MXNet Scala resources below.

## Next Steps
* [Scala API](http://mxnet.io/api/scala/)
* [More Scala Examples](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples)
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
