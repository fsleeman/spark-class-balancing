# spark-class-balancing
This project includes implementions of the following oversampling methods written in Scala for the Apache Spark platform:
* ADASYN
* ANS
* Borderline SMOTE
* CCR
* Cluster SMOTE
* Gaussian SMOTE
* _k_-Means SMOTE
* MWMOTE
* NRAS
* Random Oversampling
* RBO
* Safe Level SMOTE
* SMOTE
* SMOTE-D

Since each of these algorithms were designed for binary class problems, this Spark implementation has added multi-class support. Each minority class is processed separately but for algorithms that take in account the majority class, a one-vs-rest approach was used. In this case, all examples not part of the minority class was treated as if they were part of a single majority class. The sampling methods were developed in the style of the Spark Machine Learning Library (MLlib) methods using the _fit_ and _transform_ pattern. Each of these oversampling methods were originally presented as serial algorithms which required some modifications when implemented for Spark. However, we have kept these implementation as close to the original as possible and have not made major improvements on performance or accuracy while comparing these algorithms. Going forward, these sampling methods can be updated to better adapt to the MapReduce paradigm used by Spark.

The SamplingExample.scala file gives an example of how to load data from a csv and perform minority class oversampling. In addition, the SamplingTemplate.scala file can be used to as a template when implementing new sampling methods.

# Dependencies
Many of these sampling algorithms, such as SMOTE variants, include a _k_-Nearest Neighbors search. Instead of creating that method from scratch, an efficient implementation from spark-knn (https://github.com/saurfang/spark-knn) was used. However, the spark-knn implementation did not support nearest neighbor search by distance radius which was required by ANS and CCR. To address this limitation, spark-knn was forked (https://github.com/fsleeman/spark-knn) and the distance search feature was added. 

The spark-class-balancing library was developed using Spark 3.0.1 with Scala 2.12.1 and currently has following dependencies: spark-core, spark-sql, spark-mllib, breeze-natives, breeze, spark-knn (from the fsleeman). The original spark-knn implementation was based on Spark 2 and so the forked version was updated to be compatible with Spark 3.

# Compiling the Project
The spark-class-balancing library is designed to build a fat jar using the _sbt_ tool, with the following commands:

	sbt compile
	sbt assembly

Since the spark-knn (fsleeman fork) library is a dependency of spark-class-balancing, it must also be built as far jar using a similar method and the file must be included using the `SPARK_KNN` environmental variable which is referenced in the _build.sbt_ file. The resulting spark-class-balancing jar file can then be used for running a Spark job, both in the local or cluster mode.

# Running the Example Program
Using the local mode, the example program can be run with the following command: 

	spark-submit --class org.apache.spark.ml.sampling.SamplingExample [path to your jar file]

The run command will be different in the cluster mode and will be dependent of your specific setup.	

	

# Using Sampling Methods
Invoking one of the oversampling methods is straightforward as shown below in this example using standard SMOTE. 

  	val method = new SMOTE
  	val model = method.fit(trainData).setK(5)
  	val sampledData = model.transform(trainData)

Using the _fit/transform_ pattern, a new SMOTE estimator is instantiated and then a model is created with the _fit_ function. The resulting model is set with prediction parameters, in this case setting the _k_ value to 5, and finally the trainData DataFrame is transformed to produce oversampled data. The same process is done for every other oversampling methods, although available parameters may change.

By default, each method will attempt to oversample the minority classes to be the same size as the largest (majority) class. However, the final class counts may be slightly different because of how some underlying functions work in Spark. There are a few cases where there is still a significant amount of class imbalance because of some built in rules of the original algorithms. These oversampling methods can be updated to account for this and work better with Spark. 

In addition to majority size oversampling, you can also manually specify the oversampling rates for each minority class. The _setSamplingRatios_ function can be used on most of the methods (not CCR which modifies the majority class as well) and pass in a dictonary style _Map_ object with user specified rates. For example, If you want to oversample classes 1 and 5 by 3x and 10x respectively, use `val samplingMap: Map[String, Double] = Map( "1" -> 3.0, "5" -> 10.0 )` with
	
	val r = new SMOTE().setSamplingRatios(samplingMap)
	
The `setOversamplesOnly` function can also be used to return oversamples only, not the original and oversampled examples. This function takes a boolean for input with the default set to `false`.

