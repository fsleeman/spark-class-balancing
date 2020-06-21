package org.apache.spark.ml.sampling

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random

/*
          """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): number of neighbors in SMOTE
            n_clusters (int): number of clusters
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

        X_min= X[y == self.minority_label]

        # determining the number of samples to generate
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])

        if num_to_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        kmeans= KMeans(n_clusters= min([len(X_min), self.n_clusters]), n_jobs= self.n_jobs, random_state= self.random_state)
        kmeans.fit(X_min)
        cluster_labels= kmeans.labels_
        unique_labels= np.unique(cluster_labels)

        # creating nearest neighbors objects for each cluster
        cluster_indices= [np.where(cluster_labels == c)[0] for c in unique_labels]
        cluster_nns= [NearestNeighbors(n_neighbors= min([self.n_neighbors, len(cluster_indices[idx])])).fit(X_min[cluster_indices[idx]]) for idx in range(len(cluster_indices))]

        if max([len(c) for c in cluster_indices]) <= 1:
            _logger.info(self.__class__.__name__ + ": " + "All clusters contain 1 element")
            return X.copy(), y.copy()

        # generating the samples
        samples= []
        while len(samples) < num_to_sample:
            cluster_idx= self.random_state.randint(len(cluster_indices))
            if len(cluster_indices[cluster_idx]) <= 1:
                continue
            random_idx= self.random_state.randint(len(cluster_indices[cluster_idx]))
            sample_a= X_min[cluster_indices[cluster_idx]][random_idx]
            dist, indices= cluster_nns[cluster_idx].kneighbors(sample_a.reshape(1, -1))
            sample_b_idx= self.random_state.choice(cluster_indices[cluster_idx][indices[0][1:]])
            sample_b= X_min[sample_b_idx]
            samples.append(self.sample_between_points(sample_a, sample_b))

 */



class cluster_SMOTE {
  val knnK = 5
  var knnClusters: Array[Array[Row]] = Array[Array[Row]]()
  var knnClusterCounts: Array[Int] = Array[Int]()

  def createSample(clusterId: Int): DenseVector ={
    val row = knnClusters(clusterId)(Random.nextInt(knnClusterCounts(clusterId)))
    val features = row(1).asInstanceOf[mutable.WrappedArray[DenseVector]]

    val aSample = features(0).toArray
    val bSample = features(Random.nextInt(knnK + 1)).toArray
    val offset = Random.nextDouble()

    Vectors.dense(Array(aSample, bSample).transpose.map(x=>x(0) + offset * (x(1)-x(0)))).toDense
  }

  def calculateKnnByCluster(spark: SparkSession, df: DataFrame): DataFrame ={
    df.show()
    import spark.implicits._

      val leafSize = 10 // FIXME
      val model = new KNN().setFeaturesCol("features")
        .setTopTreeSize(df.count().toInt / 8) /// FIXME - check?
        .setTopTreeLeafSize(leafSize)
        .setSubTreeLeafSize(leafSize)
                .setK(knnK + 1) // include self example
        .setAuxCols(Array("label", "features"))
      println(model.getBalanceThreshold)
      println(model.getBufferSize)

      if(model.getBufferSize < 0.0) {
        val model = new KNN().setFeaturesCol("features")
          .setTopTreeSize(df.count().toInt / 8) /// FIXME - check?
          .setTopTreeLeafSize(leafSize)
          .setSubTreeLeafSize(leafSize)
          .setBalanceThreshold(0.0) // Fixes issue with smaller clusters
          .setK(knnK + 1) // include self example
          .setAuxCols(Array("label", "features"))
        val f = model.fit(df)
        f.transform(df).withColumn("neighborFeatures", $"neighbors.features")
      } else {
        val f = model.fit(df)
        f.transform(df).withColumn("neighborFeatures", $"neighbors.features")
      }
  }

  def fit(spark: SparkSession, dfIn: DataFrame, clusterK: Int): DataFrame = {
    val df = dfIn
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount

    println("Samples to add: " + samplesToAdd)

    val minorityDF = df.filter(df("label")===minClassLabel)

    val kValue = Math.min(minClassCount, clusterK)
    val kmeans = new KMeans().setK(kValue).setSeed(1L) // FIXME - fix seed
    val model = kmeans.fit(minorityDF)
    val predictions = model.transform(minorityDF)

    val clusters = (0 until clusterK).map(x=>predictions.filter(predictions("prediction")===x)).toArray

    // knn for each cluster
    knnClusters =  clusters.map(x=>calculateKnnByCluster(spark, x).select("label", "neighborFeatures").collect)
    knnClusterCounts = knnClusters.map(x=>x.length)

    val randomIndicies = (0 until samplesToAdd).map(_ => Random.nextInt(clusterK))
    val addedSamples = randomIndicies.map(x=>(0.toLong, minClassLabel.toInt, createSample(x))).toArray


    val dfAddedSamples = spark.createDataFrame(spark.sparkContext.parallelize(addedSamples))
      .withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    df.printSchema()
    dfAddedSamples.printSchema()
    df.union(dfAddedSamples)
  }
}
