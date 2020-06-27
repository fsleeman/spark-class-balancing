package org.apache.spark.ml.sampling

import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.sampling.utils.pointDifference

import scala.collection.mutable

/*

    """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            k (int): number of neighbors in nearest neighbors component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """


    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])

        if num_to_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min= X[y == self.minority_label]

        # fitting nearest neighbors model
        nn= NearestNeighbors(n_neighbors= min([len(X_min), self.k+1]), n_jobs= self.n_jobs)
        nn.fit(X_min)
        dist, ind= nn.kneighbors(X_min)

        # extracting standard deviations of distances
        stds= np.std(dist[:,1:], axis= 1)
        # estimating sampling density
        if np.sum(stds) > 0:
            p_i= stds/np.sum(stds)
        else:
            _logger.warning(self.__class__.__name__ + ": " + "zero distribution")
            return X.copy(), y.copy()

        # the other component of sampling density
        p_ij= dist[:,1:]/np.sum(dist[:,1:], axis= 1)[:,None]

        # number of samples to generate between minority points
        counts_ij= num_to_sample*p_i[:,None]*p_ij

        # do the sampling
        samples= []
        for i in range(len(p_i)):
            for j in range(min([len(X_min)-1, self.k])):
                while counts_ij[i][j] > 0:
                    if self.random_state.random_sample() < counts_ij[i][j]:
                        samples.append(X_min[i] + (X_min[ind[i][j+1]] - X_min[i])/(counts_ij[i][j]+1))
                    counts_ij[i][j]= counts_ij[i][j] - 1

        if len(samples) > 0:
            return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
        else:
            return X.copy(), y.copy()

 */


class SMOTE_D {
  val knnK = 5

  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    import spark.implicits._

    val df = dfIn.filter((dfIn("label") === 1) || (dfIn("label") === 5)) // FIXME
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    val minorityDF = df.filter(df("label") === minClassLabel)

    val leafSize = 10 // FIXME

    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(minorityDF.count().toInt / 8) /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(knnK + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f = model.fit(minorityDF)
    val neighbors = f.transform(minorityDF).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")
    neighbors.show()
    neighbors.printSchema()
    // udf for distances

    val calculateDistances = udf((neighbors: scala.collection.mutable.WrappedArray[DenseVector]) => {
      (1 until neighbors.length).map(x=>pointDifference(neighbors(0).toArray, neighbors(x).toArray))
    })

    val calculateStd = udf((distances: scala.collection.mutable.WrappedArray[Double]) => {
      val mean = distances.sum / distances.length.toDouble
      Math.sqrt(distances.map(x=>Math.pow(x - mean,2)).sum * (1 / (distances.length - 1).toDouble)) /// FIXME - check formula
    })

    val calculateLocalDistanceWeights = udf((distances: scala.collection.mutable.WrappedArray[Double]) => {
      distances.map(x=>x/distances.sum)
    })

    val calculateSamplesToAdd = udf((std: Double, distances: scala.collection.mutable.WrappedArray[Double]) => {
       //distances.indices.map(x=>(std * distances(x) * samplesToAdd))
      (std * minClassCount.toDouble).toInt
    })

    /*val sampleExistingExample =  udf((numberToAdd: Int, distances: scala.collection.mutable.WrappedArray[Double],
                                      neighbors: scala.collection.mutable.WrappedArray[DenseVector]) => {

      val counts = distances.map(x=>(((x + 0.5)/ distances.sum) * numberToAdd).toInt).reverse
      val originalExample = neighbors.head.toArray
      val reverseNeighbors = neighbors.tail.reverse.map(x=>x.toArray)

      // FIXME - only samples upto the required count



      def addOffset(x: Array[Double], distanceLine: Array[Double], offset: Double): Array[Double] = {
        Array[Array[Double]](x, distanceLine).transpose.map(x=>x(0) + x(1)*offset)
      }

      def getNeighborSamples(index: Int)= {
        val distanceLine: Array[Double] = Array[Array[Double]](originalExample, reverseNeighbors(index)).transpose.map(x=>x(1)-x(0))
        (0 until counts(index)).map(x=>addOffset(originalExample, distanceLine, x/counts(index).toDouble))
      }


      val x: Seq[IndexedSeq[Array[Double]]] = reverseNeighbors.indices.map(x=>getNeighborSamples(x))
      x.reduce(_ union _).toArray
    })*/

    def sampleExistingExample(numberToAdd: Int, distances: scala.collection.mutable.WrappedArray[Double],
                                      neighbors: scala.collection.mutable.WrappedArray[DenseVector]) ={

      val counts = distances.map(x=>(((x + 0.5)/ distances.sum) * numberToAdd).toInt).reverse
      val originalExample = neighbors.head.toArray
      val reverseNeighbors = neighbors.tail.reverse.map(x=>x.toArray)

      // FIXME - only samples upto the required count



      def addOffset(x: Array[Double], distanceLine: Array[Double], offset: Double): Array[Double] = {
        Array[Array[Double]](x, distanceLine).transpose.map(x=>x(0) + x(1)*offset)
      }

      def getNeighborSamples(index: Int)= {
        val distanceLine: Array[Double] = Array[Array[Double]](originalExample, reverseNeighbors(index)).transpose.map(x=>x(1)-x(0))
        (0 until counts(index)).map(x=>addOffset(originalExample, distanceLine, x/counts(index).toDouble))
      }


      val x: Seq[IndexedSeq[Array[Double]]] = reverseNeighbors.indices.map(x=>getNeighborSamples(x))
      x.reduce(_ union _).toArray
    }


    val distances = neighbors.withColumn("distances", calculateDistances(neighbors("neighborFeatures")))
    distances.show

    val std = distances.withColumn("std", calculateStd(distances("distances")))
    std.show

    val stdSum = std.select("std").collect.map(x=>x(0).toString.toDouble).sum

    val stdWeights = std.withColumn("stdWeights", std("std")/stdSum)
    stdWeights.show

    val localDistanceWeights = stdWeights.withColumn("localDistanceWeights", calculateLocalDistanceWeights(stdWeights("distances")))
    localDistanceWeights.show

    val samplesToAddDF = localDistanceWeights.withColumn("samplesToAdd", calculateSamplesToAdd(localDistanceWeights("stdWeights"), localDistanceWeights("localDistanceWeights")))
    samplesToAddDF.show
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.expressions.Window
    import org.apache.spark.sql.functions._
    val calcSamplesToAdd = samplesToAddDF.select("samplesToAdd").collect.map(x=>x(0).toString.toDouble).sum

    val sortDF = samplesToAddDF.sort(col("samplesToAdd").desc)
    sortDF.show
    println(calcSamplesToAdd)

    val partitionWindow = Window.partitionBy($"label").orderBy($"samplesToAdd".desc)
    val sumTest = sum($"samplesToAdd").over(partitionWindow)
    val runningTotals = samplesToAddDF.select($"*", sumTest as "running_total")

    val filteredTotals = runningTotals.filter(runningTotals("running_total") < samplesToAdd)

    //val addedSamples: Array[Row] = filteredTotals.withColumn("added", sampleExistingExample(filteredTotals("samplesToAdd"), filteredTotals("distances"), filteredTotals("neighborFeatures"))).select("added").collect
    val addedSamples: Array[Array[Array[Double]]] = filteredTotals.collect.map(x=>sampleExistingExample(x(8).asInstanceOf[Int], x(4).asInstanceOf[mutable.WrappedArray[Double]], x(3).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val collectedSamples = addedSamples.reduce(_ union _).map(x=>Vectors.dense(x).toDense).map(x=>Row(0.toLong, minClassLabel.toInt, x.asInstanceOf[DenseVector]))

    println(collectedSamples.length)
    /*println(collectedSamples(0).length)
    for(x<-collectedSamples(0)) {
      print(x + " ")
    }
    println("")
    println(collectedSamples(1).length)
    println(collectedSamples(2).length)
*/

    val foo: Array[(Long, Int, DenseVector)] = collectedSamples.map(x=>x.toSeq).map(x=>(x.head.toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector]))
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    dfIn.union(bar2)

  }

}
