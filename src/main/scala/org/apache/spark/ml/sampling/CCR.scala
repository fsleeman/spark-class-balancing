package org.apache.spark.ml.sampling

import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable

/*
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

        def taxicab_sample(n, r): /// FIXME - wut
            sample = []
            random_numbers= self.random_state.rand(n)

            for i in range(n):
                #spread = r - np.sum(np.abs(sample))
                spread= r
                if len(sample) > 0:
                    spread-= abs(sample[-1])
                sample.append(spread * (2 * random_numbers[i] - 1))

            return self.random_state.permutation(sample)

        minority= X[y == self.minority_label]
        majority= X[y == self.majority_label]

        energy = self.energy * (X.shape[1] ** self.scaling)

        distances= pairwise_distances(minority, majority, metric='l1')

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for i in range(len(minority)):
            minority_point= minority[i]
            remaining_energy= energy
            r= 0.0
            sorted_distances= np.argsort(distances[i])
            current_majority= 0

            while True:
                if current_majority > len(majority):
                    break

                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change= remaining_energy / (current_majority + 1.0)
                    else:
                        radius_change= remaining_energy / current_majority

                    r+= radius_change
                    break

                radius_change= remaining_energy / (current_majority + 1.0)

                if distances[i, sorted_distances[current_majority]] >= r + radius_change:
                    r+= radius_change
                    break
                else:
                    if current_majority == 0:
                        last_distance= 0.0
                    else:
                        last_distance= distances[i, sorted_distances[current_majority - 1]]

                    radius_change= distances[i, sorted_distances[current_majority]] - last_distance
                    r+= radius_change
                    remaining_energy-= radius_change * (current_majority + 1.0)
                    current_majority+= 1

            radii[i] = r

            for j in range(current_majority):
                majority_point= majority[sorted_distances[j]].astype(float)
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:
                    majority_point+= (1e-6 * self.random_state.rand(len(majority_point)) + 1e-6) * self.random_state.choice([-1.0, 1.0], len(majority_point))
                    d = np.sum(np.abs(minority_point - majority_point))

                translation = (r - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        majority= majority.astype(float)
        majority += translations

        appended= []
        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * num_to_sample))
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point + taxicab_sample(len(minority_point), r))

        if len(appended) == 0:
            _logger.info("No samples were added")
            return X.copy(), y.copy()


 */



class CCR {

  type Element = (Int, Array[Double])
  type Element2 = (Long, Int, Array[Double])


  def getManhattanDistance(example: Array[Double], neighbor: Array[Double]): Double ={
    Array(example, neighbor).transpose.map(x=>Math.abs(x(0)-x(1))).sum
  }

  val moveMajorityPoints2: UserDefinedFunction = udf((features: DenseVector, neighborIndices: mutable.WrappedArray[Long],
                                                     neighborLabels: mutable.WrappedArray[Int], neighborFeatures: mutable.WrappedArray[DenseVector], distanceArray: mutable.WrappedArray[Double], ri: Double) => {

    val majorityIndicies: Array[Long] = neighborIndices.toArray
    val majorityLabels: Array[Int] = neighborLabels.toArray
    val majorityNeighbors: Array[Array[Double]] = neighborFeatures.toArray.map(x=>x.toArray)

    // val neighborDistances: Array[Double] = distanceArray.toArray
    val distances = distanceArray.toArray

    def pointDistance(features: Array[Double], neighbor: Array[Double]): Double ={
      Array(features, neighbor).transpose.map(x=>Math.abs(x(0) - x(1))).sum
    }

    type MajorityPoint = (Long, Int, Array[Double])

    def getMovedNeighbors(j: Int): (Boolean, (Long, Int, Array[Double])) ={
       println("^^^ " + distances(j) + " " + ri)
      if(distances(j) <= ri) {
        // println("@@", distances(j), ri)
        val d = pointDistance(features.toArray, majorityNeighbors(j))
        // FIXME - check line 19 in algorithm for tj usage
        val scale =  (ri - d) / d
        val offset: Array[Double] = Array(features.toArray, majorityNeighbors(j)).transpose.map(x=>x(0) - x(1)).map(x=>x * scale)
        val updatedPosition = Array(offset, majorityNeighbors(j)).transpose.map(x=>x(0)+x(1))
        (true, (majorityIndicies(j), majorityLabels(j), updatedPosition))
      } else {
        (false, (majorityIndicies(j), majorityLabels(j), majorityNeighbors(j)))
      }
    }

    println("~~~ indicies: " + majorityNeighbors.indices.length)
    val movedMajoirtyNeigbors = majorityNeighbors.indices.map(j=>getMovedNeighbors(j)).filter(x=>x._1).map(x=>x._2)
    //println(movedMajoirtyNeigbors.length)
    //println(movedMajoirtyNeigbors(0))
    //val combinedMajoritNeighbors = movedMajoirtyNeigbors.reduce(_ union _)

    //val xxx = combinedMajoritNeighbors.filter(x=>x._1)
    //print(xxx.length)
    //for(x<-xxx) {
     // println(x)
   // }

    //print("***** " + combinedMajoritNeighbors.length)

    //combinedMajoritNeighbors


    movedMajoirtyNeigbors
  })

  val moveMajorityPoints: UserDefinedFunction = udf((features: DenseVector, neighborIndices: mutable.WrappedArray[Long],
                                        neighborLabels: mutable.WrappedArray[Int], neighborFeatures: mutable.WrappedArray[DenseVector], distanceArray: mutable.WrappedArray[Double], ri: Double) => {
    val majorityIndicies: Array[Long] = neighborIndices.toArray
    val majorityLabels: Array[Int] = neighborLabels.toArray
    val majorityNeighbors: Array[Array[Double]] = neighborFeatures.toArray.map(x=>x.toArray)

    // val neighborDistances: Array[Double] = distanceArray.toArray
    val distances = distanceArray.toArray

    def pointDistance(features: Array[Double], neighbor: Array[Double]): Double ={
      Array(features, neighbor).transpose.map(x=>Math.abs(x(0) - x(1))).sum
    }

    type MajorityPoint = (Long, Int, Array[Double])

    def getMovedNeighbors(j: Int): Array[(Long, Int, Array[Double])] ={
      // println("^^^ " + distances(j) + " " + ri)
      if(distances(j) <= ri) {
        // println("@@", distances(j), ri)
        val d = pointDistance(features.toArray, majorityNeighbors(j))
        // FIXME - check line 19 in algorithm for tj usage
        val scale =  (ri - d) / d
        val offset: Array[Double] = Array(features.toArray, majorityNeighbors(j)).transpose.map(x=>x(0) - x(1)).map(x=>x * scale)
        val updatedPosition = Array(offset, majorityNeighbors(j)).transpose.map(x=>x(0)+x(1))

        Array[(Long, Int, Array[Double])]((majorityIndicies(j), majorityLabels(j), updatedPosition))
      } else {
        Array[(Long, Int, Array[Double])]()
      }
    }

    println("~~~ indicies: " + majorityNeighbors.indices.length)
    val movedMajoirtyNeigbors = majorityNeighbors.indices.map(j=>getMovedNeighbors(j))
    val combinedMajoritNeighbors: Array[(Long, Int, Array[Double])] = movedMajoirtyNeigbors.reduce(_ union _)
    print(combinedMajoritNeighbors(0))
    print("***** " + combinedMajoritNeighbors.length)

    combinedMajoritNeighbors
    // return cleaning radius and moved majority points

  })

  val stuff: UserDefinedFunction = udf((features: DenseVector, distanceArray: mutable.WrappedArray[Double]) => {
    // val majorityIndicies: Array[Long] = neighborIndices.toArray
    // val majorityLabels: Array[Int] = neighborLabels.toArray
    // val majorityNeighbors: Array[Array[Double]] = neighborFeatures.toArray.map(x=>x.toArray)

    // val neighborDistances: Array[Double] = distanceArray.toArray
    val distances = distanceArray.toArray

    var energyBudget = 0.64
    var ri = 0.0
    var deltaR = energyBudget


    def nearestNoWithinR(distances: Array[Double], r: Double): Double ={

      def setWithinValue(d: Double, r: Double): Double ={
        if(d < r) {
          Double.MaxValue
        } else {
          d
        }
      }
      distances.map(x=>setWithinValue(x, r)).min
    }
    def NoP(distances: Array[Double], radius: Double): Int = {
      def isWithinRadius(d: Double): Int ={
        if (d <= radius) {
          1
        } else {
          0
        }
      }

      distances.map(x=>isWithinRadius(x)).sum + 1
    }

    // generate cleaning radius
    while(energyBudget > 0.0) {
      val NoPValue = NoP(distances, ri)
      deltaR = energyBudget / NoPValue.toDouble
      if(NoP(distances, ri + deltaR) > NoPValue) {
        deltaR = nearestNoWithinR(distances, ri)
      }
      ri = ri + deltaR
      energyBudget = energyBudget - deltaR * NoPValue
    }
    Math.pow(ri, -1)

    // val updatedMajorityRows = Array[Row]()
/*

    def getMovedNeighbors(j: Int): Array[Row] ={
      // println("^^^ " + distances(j) + " " + ri)
      if(distances(j) <= ri) {
        // println("@@", distances(j), ri)
        val d = pointDistance(features.toArray, majorityNeighbors(j))
        // FIXME - check line 19 in algorithm for tj usage
        val scale =  (ri - d) / d
        val offset: Array[Double] = Array(features.toArray, majorityNeighbors(j)).transpose.map(x=>x(0) - x(1)).map(x=>x * scale)
        val updatedPosition = Array(offset, majorityNeighbors(j)).transpose.map(x=>x(0)+x(1))

        Array[Row](Row((majorityIndicies(j), majorityLabels(j), updatedPosition)))
      } else {
        Array[Row]()
      }
    }

    println("~~~ indicies: " + majorityNeighbors.indices.length)
    val movedMajoirtyNeigbors = majorityNeighbors.indices.map(j=>getMovedNeighbors(j))
    val combinedMajoritNeighbors: Array[Row] = movedMajoirtyNeigbors.reduce(_ union _)
    print(combinedMajoritNeighbors(0))
    print("***** " + combinedMajoritNeighbors.length)


    // return cleaning radius and moved majority points
  */
  })


  def extractMovedPoints(index: Array[Long], label: Array[Int], feature: Array[Array[Double]]): Array[Row] ={
    val X = index.indices.map(x=>Row(index(x), label(x), feature(x))).toArray
    X
  }

  def createSyntheicPoints(row: Row): Array[Row] ={
    val label = row(0).toString
    val features = row(1).asInstanceOf[DenseVector].toArray
    val r = row(2).toString.toDouble
    val examplesToAdd = Math.floor(row(3).toString.toDouble + 0.5).toInt

    val random = scala.util.Random
    // (0 until examplesToAdd).map(_=>Row(0L, label, for(f <- features) yield Vectors.dense(f * (random.nextDouble() * 2.0 - 1) * r))).toArray
    (0 until examplesToAdd).map(_=>Row(0L, label, Vectors.dense(for(f <- features) yield f * (random.nextDouble() * 2.0 - 1) * r))).toArray
  }


  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    import spark.implicits._

    // parameters
    // proportion = 1.0, energy = 1.0, scaling = 0.0

    val df = dfIn.filter((dfIn("label") === 1) || (dfIn("label") === 5)) // FIXME
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    val minorityDF = df.filter(df("label") === minClassLabel)
    val majorityDF = df.filter(df("label") === maxClassLabel)

    val leafSize = 100
    val kValue = 10 /// FIXME - switch to distance?

    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("index", "label", "features"))
      .setQueryByDistance(false)

    println(model.getQueryMode)

    val f: KNNModel = model.fit(majorityDF)
    f.setDistanceCol("distances")

    val t = f.transform(minorityDF).sort("index")
    println("*** first knn ****")
    t.show
    t.printSchema()


    val test = t.withColumn("majorityIndex", $"neighbors.index")
      .withColumn("majorityLabel", $"neighbors.label")
      .withColumn("majorityPoints", $"neighbors.features").drop("neighbors")//.take(1)
    test.show
    test.printSchema()

    // FIXME - use full DF
    val test2 = test.take(10)

    val foo = test2.map(x=>x.toSeq).map(x=>(x.head.toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector],
      x(3).asInstanceOf[mutable.WrappedArray[Double]], x(4).asInstanceOf[mutable.WrappedArray[Long]],
      x(5).asInstanceOf[mutable.WrappedArray[Int]], x(6).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val testDF = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")
      .withColumnRenamed("_4", "distances")
      .withColumnRenamed("_5", "majorityIndex")
      .withColumnRenamed("_6", "majorityLabel")
      .withColumnRenamed("_7", "majorityFeatures")


    //println("****** at result")
    testDF.show
    testDF.printSchema()


    val result = testDF.withColumn("ri", stuff($"features", $"distances"))
    result.show
    val inverseRiSum = result.select("ri").rdd.map(x=>x(0).toString.toDouble).reduce(_ + _)
    println("inverse sum " + inverseRiSum)


    val resultWithSampleCount = result.withColumn("gi", ($"ri"/ inverseRiSum) * samplesToAdd)
    resultWithSampleCount.show

    val createdPoints: Array[Array[Row]] = resultWithSampleCount.drop("index", "distances", "majorityIndex",
      "majorityLabel", "majorityFeatures").collect().map(x=>createSyntheicPoints(x))
    println("***")
    for(x<-createdPoints) {
      println(x.length)
    }

    val unionedPoints = createdPoints.reduce(_ union _)
    println("~~~~~~ oversampled points: " + unionedPoints.length)

    val movedPoints = resultWithSampleCount.withColumn("movedMajorityPoints",
      moveMajorityPoints2($"features",  $"majorityIndex",  $"majorityLabel", $"majorityFeatures", $"distances", $"ri"))
    movedPoints.show()
    movedPoints.printSchema()

    val movedPointsExpanded = movedPoints.withColumn("movedMajorityIndex", $"movedMajorityPoints._1")
      .withColumn("movedMajorityLabel", $"movedMajorityPoints._2")
      .withColumn("movedMajorityExamples", $"movedMajorityPoints._3")
      .drop("movedMajorityPoints")


    val movedPointsSelected = movedPointsExpanded.select("movedMajorityIndex", "movedMajorityLabel", "movedMajorityExamples")
    movedPointsSelected.show()

    val movedPointsCollected = movedPointsSelected.collect()

    val fooX = movedPointsCollected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Long]].toArray,
      x(1).asInstanceOf[mutable.WrappedArray[Int]].toArray,
      x(2).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Double]]].toArray.map(y=>y.toArray)))

    val results = fooX.map(x=>extractMovedPoints(x._1, x._2, x._3))

    val total = results.reduce(_ union _)
    println(total.length)
    /*for(x <- total.indices) {
      println(total(x)(0))
    }*/

    // val totalIndicies = total.map(x=>x(0))

    val grouped: Map[Long, Array[Row]] = total groupBy (s => s(0).toString.toLong)


    def getAveragedRow(rows: Array[Row]): Row ={
      val data: Array[Double] = rows.map(x=>x(2).asInstanceOf[Array[Double]]).transpose.map(x=>x.sum)
      Row(rows(0)(0).toString.toLong, rows(0)(1).toString.toInt, Vectors.dense(data))


    }

    val averaged: Array[Row] = grouped.map(x=>getAveragedRow(x._2)).toArray

    val movedMajorityIndicies = averaged.map(x=>x(0).toString.toLong).toList
    /*for(x<-movedMajorityIndicies) {
      println(x)
    }

    println("~~~~~")
    println(df.count)


    df.printSchema()*/



    println("############")
    println(averaged(0))
    println(unionedPoints(0))


    val movedMajorityExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(averaged.map(x=>(x(0).toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector])))).toDF() // x(2).asInstanceOf[Array[Double]])))).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2","label")
      .withColumnRenamed("_3","features")

    movedMajorityExamplesDF.show
    movedMajorityExamplesDF.printSchema()
    println(movedMajorityExamplesDF.count())


    val syntheticExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(unionedPoints.map(x=>(x(0).toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector])))).toDF() // x(2).asInstanceOf[Array[Double]])))).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2","label")
      .withColumnRenamed("_3","features")

    syntheticExamplesDF.show
    syntheticExamplesDF.printSchema()
    println(syntheticExamplesDF.count)



    val keptMajorityDF = df.filter(!$"index".isin(movedMajorityIndicies: _*))
    println(keptMajorityDF.count)
    keptMajorityDF.show()
    keptMajorityDF.printSchema()
    println(keptMajorityDF.count())


    val finalDF = keptMajorityDF.union(movedMajorityExamplesDF).union(syntheticExamplesDF)

    finalDF.show()
    println(finalDF.count())


    /*// val df = dfIn.filter((dfIn("label") === 1) || (dfIn("label") === 5)) // FIXME
    val values = Array(2708, 2794)
    val xxx: Array[Row] = averaged.filter(x=>values.contains(x(0)))




    for(x<-averaged) {
      println(x)
    }
    println("***************")
    for(x<-xxx) {
      println(x)
    }*/

    /*val averagedPoints = for((key, value) <- grouped) {

    }*/

            /*for((key,language) <- grouped){
          println(count + " -> " + language)

        }

        val taken = grouped.keys.foreach( (s) =>
          println( "x" + s)

        )*/



//  def extractMovedPoints(indices: Array[Long], labels: Array[Int], features: Array[Array[Double]]): Int ={

    /*val fooX = movedPointsCollected.map(x=>x(0).asInstanceOf[mutable.WrappedArray[Long]].toArray)
    println(fooX(0).toVector)

    val fooXX = movedPointsCollected.map(x=>x(1).asInstanceOf[mutable.WrappedArray[Int]].toArray)
    println(fooXX(0).toVector)

    val fooXXX = movedPointsCollected.map(x=>x(2).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Double]]].toArray.map(y=>y.toArray.toVector))
    println(fooXXX(0)(0))*/

    //val barX = fooX(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Double]]].toArray
    //val barY = barX.map(x=>x.toArray.toVector)
    //println(barY(0))


    //val xx = movedPoints.select("movedMajorityPoints").collect()
    //val foo2 = xx.map(x=>x(0))
    //println(foo2(0).asInstanceOf[mutable.WrappedArray[(Long, Int, mutable.WrappedArray[Double])]])

    finalDF
  }

}
