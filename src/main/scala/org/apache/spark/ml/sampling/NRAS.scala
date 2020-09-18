package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.{Element, getCountsByClass}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, lit, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.knn.{KNN, KNNModel}
import utils.getMatchingClassCount
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.ml.linalg.{DenseVector, Vectors}

import scala.collection.mutable
import scala.util.Random

/*

 """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): number of neighbors
            t (float): [0,1] fraction of n_neighbors as threshold
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

        # standardization is needed to make the range of the propensity scores similar to that of the features
        mms= MinMaxScaler()
        X_trans= mms.fit_transform(X)

        # determining propensity scores using logistic regression
        lr= LogisticRegression(solver= 'lbfgs', n_jobs= self.n_jobs, random_state= self.random_state)
        lr.fit(X_trans, y)
        propensity= lr.predict_proba(X_trans)[:,np.where(lr.classes_ == self.minority_label)[0][0]]

        X_min= X_trans[y == self.minority_label]

        # adding propensity scores as a new feature
        X_new= np.column_stack([X_trans, propensity])
        X_min_new= X_new[y == self.minority_label]

        # finding nearest neighbors of minority samples
        n_neighbors= min([len(X_new), self.n_neighbors+1])
        nn= NearestNeighbors(n_neighbors= n_neighbors, n_jobs= self.n_jobs)
        nn.fit(X_new)
        dist, ind= nn.kneighbors(X_min_new)

        # do the sampling
        samples= []
        to_remove= []
        while len(samples) < num_to_sample:
            idx= self.random_state.randint(len(X_min))
            # finding the number of minority neighbors
            t_hat= np.sum(y[ind[idx][1:]] == self.minority_label)
            if t_hat < self.t*n_neighbors:
                # removing the minority point if the number of minority neighbors is less then the threshold
                # to_remove indexes X_min
                if not idx in to_remove:
                    to_remove.append(idx)
                    # compensating the removal of the minority point
                    num_to_sample= num_to_sample + 1

                if len(to_remove) == len(X_min):
                    _logger.warning(self.__class__.__name__ + ": " +"all minority samples identified as noise")
                    return X.copy(), y.copy()
            else:
                # otherwise do the sampling
                samples.append(self.sample_between_points(X_min[idx], X_trans[self.random_state.choice(ind[idx][1:])]))




 */




/** Transformer Parameters*/
private[ml] trait NRASModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class NRASModel private[ml](override val uid: String) extends Model[NRASModel] with NRASModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  // FIXME - reuse from basic SMOTE?
  def getSmoteSample(row: Row): Row = {
    val index = row(0).toString.toLong
    val label = row(1).toString.toInt
    val features: Array[Double] = row(2).asInstanceOf[DenseVector].toArray
    val neighbors = row(3).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val randomNeighbor: Array[Double] = neighbors(Random.nextInt(neighbors.length)).toArray

    val gap = randomNeighbor.indices.map(_=>Random.nextDouble()).toArray // FIXME - should this be one value instead?

    val syntheticExample = Vectors.dense(Array(features, randomNeighbor, gap).transpose.map(x=>x(0) + x(2) * (x(1)-x(0)))).toDense

    Row(index, label, syntheticExample)
  }

  def oversampleClass(dataset: Dataset[_], minorityClassLabel: String, samplesToAdd: Int): DataFrame = {
    println("**** samples to add: " + samplesToAdd)
    val spark = dataset.sparkSession
    import spark.implicits._
    println(dataset.take(1)(0))

    val df = dataset.toDF()

    /*val fix: UserDefinedFunction = udf((features: DenseVector) => {
      (54, features)

    })*/

    //.filter(dataset("label") === 2 || dataset("label") ===3)//.withColumn("fixed", fix(dataset("features")))

    println("label: " + minorityClassLabel)
    // val minorityDF = dataset.filter(dataset("label") === minorityClassLabel)
    // val majorityDF = dataset.filter(dataset("label") =!= minorityClassLabel)
    /*
     # determining propensity scores using logistic regression
        lr= LogisticRegression(solver= 'lbfgs', n_jobs= self.n_jobs, random_state= self.random_state)
        lr.fit(X_trans, y)
        propensity= lr.predict_proba(X_trans)[:,np.where(lr.classes_ == self.minority_label)[0][0]]
     */
    //val training = dataset.sparkSession.read.format("libsvm").load("/home/ford/Documents/sampling/sample_libsvm_data.txt")
    //training.show
    //training.printSchema()
    dataset.show()
    dataset.printSchema()
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1e-6)
      .setRegParam(0.01)



    val lrModel = lr.fit(dataset).setProbabilityCol("probability")
    //println(lrModel.explainParams())
    var labelIndex = -1

    for(x <- lrModel.summary.labels.indices) {
      if(lrModel.summary.labels(x).toInt.toString == minorityClassLabel) {
        labelIndex = x
      }
    }

    println("model label index: " + labelIndex + " cls: " + lrModel.summary.labels(labelIndex))

    val propensity = lrModel.transform(dataset)
    propensity.show()
    propensity.printSchema()

    val minorityClassProbablity = udf((probs: DenseVector, i: Int) => probs(i))
    val probs = propensity.withColumn("minorityClassProbability", minorityClassProbablity(propensity("probability"),lit(labelIndex)))
    probs.show




    val addPropensityValue = udf((features: DenseVector, probability: Double) => {
      Vectors.dense(features.toArray :+ probability).toDense
    })

    val updated = probs.withColumn("updatedFeatures", addPropensityValue(probs("features"), probs("minorityClassProbability")))
      .drop("features").withColumnRenamed("updatedFeatures", "features")
    updated.show


    val leafSize = 100
    val kValue = 5
    val model = new KNN().setFeaturesCol("features")
     .setTopTreeSize(10) /// FIXME - check?
     .setTopTreeLeafSize(leafSize)
     .setSubTreeLeafSize(leafSize)
     .setK(kValue + 1) // include self example
     .setAuxCols(Array("label", "features"))

    val f: KNNModel = model.fit(propensity).setDistanceCol("distances")

    val minorityDF = propensity.filter(propensity("label") === minorityClassLabel)
    val minorityKnn = f.transform(minorityDF)
    minorityKnn.show

    val threshold = 3
    val neighborCounts = minorityKnn.withColumn("sameClassNeighbors", getMatchingClassCount(minorityKnn("neighbors.label"), minorityKnn("label")))
    val neighborClassFiltered = neighborCounts.filter(neighborCounts("sameClassNeighbors") >= threshold)

    neighborClassFiltered.show

    val dataForSmote = neighborClassFiltered.select("index", "label", "features", "neighbors")
    dataForSmote.show
    dataForSmote.printSchema()
    val dataForSmoteCollected = dataForSmote.withColumn("neighborFeatures", $"neighbors.features").drop("neighbors").collect

    val randomIndicies = (0 until samplesToAdd).map(_=>Random.nextInt(dataForSmoteCollected.length))
    val createdSamples = spark.createDataFrame(spark.sparkContext.parallelize(randomIndicies.map(x=>getSmoteSample(dataForSmoteCollected(x)))), df.schema)

    createdSamples
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val counts = getCountsByClass(df.sparkSession, "label", df).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt


    val minorityClasses: Array[(String, Int)] = counts.collect.map(x=>(x(0).toString, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)
    //val minorityClasses: Array[(String, Int)] = counts.collect.take(1).map(x=>(x(0).toString, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)
    println("minorityClasses count: " + minorityClasses.length)
    val results: DataFrame = minorityClasses.map(x=>oversampleClass(dataset, x._1, majorityClassCount - x._2)).reduce(_ union _).union(dataset.toDF())

    println("dataset: " + dataset.count)
    println("added: " + results.count)
    getCountsByClass(df.sparkSession, "label", results)


    results
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): NRASModel = {
    val copied = new NRASModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}




/** Estimator Parameters*/
private[ml] trait NRASParams extends NRASModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class NRAS(override val uid: String) extends Estimator[NRASModel] with NRASParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): NRASModel = {
    val model = new NRASModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): NRAS = defaultCopy(extra)

}
