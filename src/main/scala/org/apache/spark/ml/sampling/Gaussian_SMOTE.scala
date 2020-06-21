package org.apache.spark.ml.sampling

import breeze.stats.distributions.Gaussian
import org.apache.hadoop.hdfs.server.datanode.fsdataset.impl.RamDiskReplicaLruTracker
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random
// import org.apache.commons.math3.analysis.function.Gaussian
/*
   """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations


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

        if self.class_stats[self.minority_label] < 2:
            _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()

        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])

        if num_to_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # standardization applied to make sigma compatible with the data
        ss= StandardScaler()
        X_ss= ss.fit_transform(X)

        # fitting nearest neighbors model to find the minority neighbors of minority samples
        X_min= X_ss[y == self.minority_label]
        nn= NearestNeighbors(n_neighbors= min([len(X_min), self.n_neighbors + 1]), n_jobs= self.n_jobs)
        nn.fit(X_min)
        dist, ind= nn.kneighbors(X_min)

        # do the sampling
        samples= []
        while len(samples) < num_to_sample:
            idx= self.random_state.randint(len(X_min))
            random_neighbor= self.random_state.choice(ind[idx][1:])
            s0= self.sample_between_points(X_min[idx], X_min[random_neighbor])
            samples.append(self.random_state.normal(s0, self.sigma))

 */


// FIXME - check normalization/scaling
class Gaussian_SMOTE {

  def getSingleDistance(x: Array[Double], y: Array[Double]): Double = {
    var distance = 0.0
    for(index<-x.indices) {
      distance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distance
  }

  def getGaussian(u: Double, s: Double): Double = {
    val g = Gaussian(u, s)
    g.draw()
  }

  def getSmoteSample(row: Row): Row = {
    val index = row(0).toString.toLong
    val label = row(1).toString.toInt
    val features = row(2).asInstanceOf[DenseVector].toArray
    val neighbors = row(3).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val randomNeighbor = neighbors(Random.nextInt(neighbors.length)).toArray

    val randmPoint = Random.nextDouble()
    val gap = features.indices.map(x => (randomNeighbor(x) - features(x)) * randmPoint).toArray

    val ranges = features.indices.map(x => getGaussian(features(x), 0.5)).toArray
    val syntheticExample = Vectors.dense(Array(features, randomNeighbor, ranges).transpose.map(x => x(0) + (x(1) - x(0) * x(2)))).toDense

    Row(index, label, syntheticExample)
  }


  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {

    import spark.implicits._

    val df = dfIn.filter((dfIn("label") === 1) || (dfIn("label") === 5)) // FIXME
    val m = 5   // k-value
    val leafSize = 1000

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    /*** For each minority example, calculate the m nn's in training set***/
    val minorityDF = df.filter(df("label")===minClassLabel)
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(m + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f = model.fit(minorityDF)
    val t = f.transform(minorityDF)

    t.show()
    t.printSchema()

    /*while len(samples) < num_to_sample:
        idx= self.random_state.randint(len(X_min))
        random_neighbor= self.random_state.choice(ind[idx][1:])
        s0 = self.sample_between_points(X_min[idx], X_min[random_neighbor])
        samples.append(self.random_state.normal(s0, self.sigma))*/

    val gauss = Gaussian(0.0, 0.5)
    println(gauss.draw())
    println(gauss.draw())
    println(gauss.draw())
    println(gauss.draw())
    println(gauss.draw())


    dfIn
  }
}
