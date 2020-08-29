package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}


/*

        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
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

        if self.class_stats[self.minority_label] < 2:
            _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()

        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])

        if num_to_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min= X[y == self.minority_label]

        # outcast extraction algorithm

        # maximum C value
        C_max= int(0.25*len(X))

        # finding the first minority neighbor of minority samples
        nn= NearestNeighbors(n_neighbors= 2, n_jobs= self.n_jobs)
        nn.fit(X_min)
        dist, ind= nn.kneighbors(X_min)

        # extracting the distances of first minority neighbors from minority samples
        first_pos_neighbor_distances= dist[:,1]

        # fitting another nearest neighbors model to extract majority samples in
        # the neighborhoods of minority samples
        nn= NearestNeighbors(n_neighbors= 1, n_jobs= self.n_jobs)
        nn.fit(X)

        # extracting the number of majority samples in the neighborhood of minority samples
        out_border= []
        for i in range(len(X_min)):
            ind= nn.radius_neighbors(X_min[i].reshape(1, -1), first_pos_neighbor_distances[i], return_distance= False)
            out_border.append(np.sum(y[ind[0]] == self.majority_label))

        out_border= np.array(out_border)

        # finding the optimal C value by comparing the number of outcast minority
        # samples when traversing the range [1, C_max]
        n_oc_m1= -1
        C= 0
        best_diff= np.inf
        for c in range(1, C_max):
            n_oc= np.sum(out_border >= c)
            if abs(n_oc - n_oc_m1) < best_diff:
                best_diff= abs(n_oc - n_oc_m1)
                C= n_oc
            n_oc_m1= n_oc

        # determining the set of minority samples Pused
        Pused= np.where(out_border < C)[0]

        # Adaptive neighbor SMOTE algorithm

        # checking if there are minority samples left
        if len(Pused) == 0:
            _logger.info(self.__class__.__name__ + ": " + "Pused is empty")
            return X.copy(), y.copy()

        # finding the maximum distances of first positive neighbors
        eps= np.max(first_pos_neighbor_distances[Pused])

        # fitting nearest neighbors model to find nearest minority samples in
        # the neighborhoods of minority samples
        nn= NearestNeighbors(n_neighbors= 1, n_jobs= self.n_jobs)
        nn.fit(X_min[Pused])
        ind= nn.radius_neighbors(X_min[Pused], eps, return_distance= False)

        # extracting the number of positive samples in the neighborhoods
        Np= np.array([len(i) for i in ind])

        if np.all(Np == 1):
            _logger.warning(self.__class__.__name__ + ": " + "all samples have only 1 neighbor in the given radius")
            return X.copy(), y.copy()

        # determining the distribution used to generate samples
        distribution= Np/np.sum(Np)

        # generating samples
        samples= []
        while len(samples) < num_to_sample:
            random_idx= self.random_state.choice(np.arange(len(Pused)), p= distribution)
            if len(ind[random_idx]) > 1:
                random_neighbor_idx= self.random_state.choice(ind[random_idx])
                while random_neighbor_idx == random_idx:
                    random_neighbor_idx= self.random_state.choice(ind[random_idx])
                samples.append(self.sample_between_points(X_min[Pused[random_idx]], X_min[Pused[random_neighbor_idx]]))

 */




/** Transformer Parameters*/
private[ml] trait ANSModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class ANSModel private[ml](override val uid: String) extends Model[ANSModel] with ANSModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))


  override def transform(dataset: Dataset[_]): DataFrame = {
    
    val df = dataset.filter((dataset("label") === 1) || (dataset("label") === 5)).toDF // FIXME
    val spark = df.sparkSession
    import spark.implicits._
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    val minorityDF = df.filter(df("label") === minClassLabel)

    val C_max= (0.25*df.count()).toInt

    val leafSize = 10 // FIXME

    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(minorityDF.count().toInt / 8) /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(1 + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f = model.fit(minorityDF)

    val neighbors = f.transform(minorityDF).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")
    neighbors.show()
    neighbors.printSchema()

    /*
    # finding the first minority neighbor of minority samples
          nn= NearestNeighbors(n_neighbors= 2, n_jobs= self.n_jobs)
          nn.fit(X_min)
          dist, ind= nn.kneighbors(X_min)
  
          # extracting the distances of first minority neighbors from minority samples
          first_pos_neighbor_distances= dist[:,1]
  
    # extracting the number of majority samples in the neighborhood of minority samples
          out_border= []
          for i in range(len(X_min)):
              ind= nn.radius_neighbors(X_min[i].reshape(1, -1), first_pos_neighbor_distances[i], return_distance= False)
              out_border.append(np.sum(y[ind[0]] == self.majority_label))
  
          out_border= np.array(out_border)
  
   */


    dataset.toDF
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): ANSModel = {
    val copied = new ANSModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait ANSParams extends ANSModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class ANS(override val uid: String) extends Estimator[ANSModel] with ANSParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): ANSModel = {
    val model = new ANSModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ANS = defaultCopy(extra)

}

