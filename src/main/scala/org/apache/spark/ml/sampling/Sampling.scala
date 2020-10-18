package org.apache.spark.ml.sampling

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.linalg.SparseVector

import scala.io.Source
import scala.util.Random
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

//FIXME - turn classes back to Ints instead of Doubles
object Sampling {

  var resultIndex = 0
  var results = ""
  var resultArray: Array[Array[String]] = Array()

  // FIXME - define this in another file?
  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)

  def maxValue(a: Double, b:Double): Double ={
    if(a >= b) { a }
    else { b }
  }
  def minValue(a: Double, b:Double): Double ={
    if(a <= b) { a }
    else { b }
  }

  def printArray(a: Array[Double]) ={
    for(x<-a) {
      print(x + " ")
    }
    println("")
  }


  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame, labels: Array[Double]): Array[String]={//String ={

    println("LABELS")
    for(x<-labels) {
      println(x)
    }
    distinctClasses.printSchema()

    //FIXME - don't calculate twice

    val classLabels = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble)

    //val maxLabel: Double = classLabels.max
    //val minLabel: Double = classLabels.min
    val numberOfClasses = classLabels.length
    //val classCount = confusionMatrix.columns.length - 1
    //val testLabels = distinctClasses.map(_.getAs[String]("label")).map(x => x.toDouble).collect().sorted

    val rows: Array[Array[Double]] = confusionMatrix.collect.map(_.toSeq.toArray.map(_.toString.toDouble))

    // val labels = Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    val classMaps = rows.indices.map(x=>(rows(x).head, x)).toMap
    println(classMaps)
    for(x<-classMaps) {
  //    println(x._1, x._2)
    }

    var updatedRows = Array[Array[Double]]()
    for(x<-labels.indices) {
      if(classMaps.contains(labels(x))) {
        // println(rows(classMaps(x)).toString)
    //    printArray(rows(classMaps(x)))
        updatedRows = updatedRows :+ rows(classMaps(x))

      } else {
        //updatedRows +: Array[Double](labels(x)) ++ labels.indices.map(_=>0.0)
        val emptyRow = Array[Double](labels(x)) ++ labels.indices.map(_=>0.0) //, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) // Array[Double](labels(x)) ++ labels.indices.map(_=>0.0)
        // println(emptyRow.toString)
      //  printArray(emptyRow)
        updatedRows = updatedRows :+ emptyRow

      }
    }

    println("updated rows")
    for(x<-updatedRows) {
      printArray(x)
    }

    println("****")

    val totalCount = updatedRows.map(x => x.tail.sum).sum
    // val classMaps: Array[(Int, Double)] = testLabels.zipWithIndex.map(x => (x._2, x._1))

    /*println("classMaps")
    for(x<-classMaps) {
      println(x._1, x._2)
    }*/

    var AvAvg = 0.0
    var MAvG = 1.0
    var RecM = 0.0
    var PrecM = 0.0
    var Precu = 0.0
    var Recu = 0.0
    var FbM = 0.0
    var Fbu = 0.0
    var AvFb = 0.0
    var CBA = 0.0

    var tSum = 0.0
    var pSum = 0.0
    var tpSum = 0.0
    val beta = 0.5 // User specified

    //FIXME - double check updated values

    //val labels = Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    //FIXME - could be made parallel w/udf
    //for (clsIndex <- minLabel to maxLabel - minLabel) {
    for (clsIndex <- labels.indices) {

      val colSum = updatedRows.map(x => x.tail(clsIndex)).sum
      // val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) updatedRows.filter(x => x.head == clsIndex)(0).tail.map(x => x).sum else 0
      val rowValueSum = updatedRows(clsIndex).tail.sum
      //println("clsIndex: " + clsIndex + " colSum: " + colSum + " rowSum: " + rowValueSum)
      // val tp: Double = if (classMaps.map(x => x._2).contains(clsIndex)) updatedRows.filter(x => x.head == clsIndex)(0).tail(clsIndex) else 0
      val tp: Double = updatedRows(clsIndex).tail(clsIndex)
      val fn: Double = colSum - tp // check
      val fp: Double = rowValueSum - tp // check
      val tn: Double = totalCount - tp - fp - fn

      println(tp + " " + tn + " " + fp + " " + fn)

      val recall = tp / (tp + fn)
      val precision = tp / (tp + fp)

      AvAvg += ((tp + tn) / (tp + tn + fp + fn))
      MAvG *= recall
      RecM += { if(recall.isNaN) 0.0 else recall }
      PrecM += precision

      val getAvFb: Double= {
        val result = ((1 + Math.pow(beta, 2.0)) * precision * recall) / (Math.pow(beta, 2.0) * precision + recall)
        if(result.isNaN) {
          0.0
        }
        else result
      }
      AvFb += getAvFb

      //FIXME - what to do if col/row sum are zero?
      val rowColMaxValue = maxValue(colSum, rowValueSum)
      if(rowColMaxValue > 0) {
        CBA += tp / rowColMaxValue
      }
      else {
        //println("CBA value NaN")
      }

      // for Recu and Precu
      tpSum += tp
      tSum += (tp + fn)
      pSum += (tp + fp)
    }

    AvAvg /= numberOfClasses
    MAvG = {  val result = Math.pow(MAvG, 1/numberOfClasses.toDouble); if(result.isNaN) 0.0 else result } //Math.pow((MAvG), (1/numberOfClasses.toDouble))
    RecM /= numberOfClasses
    PrecM /= numberOfClasses
    Recu = tpSum / tSum
    Precu = tpSum / pSum
    FbM = { val result = ((1 + Math.pow(beta, 2.0)) * PrecM * RecM) / (Math.pow(beta, 2.0) * PrecM + RecM); if(result.isNaN) 0.0 else result }
    Fbu = { val result = ((1 + Math.pow(beta, 2.0)) * Precu * Recu) / (Math.pow(beta, 2.0) * Precu + Recu); if(result.isNaN) 0.0 else result }
    AvFb /= numberOfClasses
    CBA /= numberOfClasses

    Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)
  }

  //assume there is only one class present
  def overSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]() //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
    //val samplesToAdd = numSamples - df.count()
    val currentCount = df.count()
    if (0 < currentCount && currentCount < numSamples) {
      val currentSamples = df.sample(withReplacement = true, (numSamples - currentCount) / currentCount.toDouble).collect()
      samples = samples ++ currentSamples
    }

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)
    df.union(x).toDF()
  }

  def underSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]() //FIXME - make this more parallel

    val underSampleRatio = numSamples / df.count().toDouble
    if (underSampleRatio < 1.0) {
      val currentSamples = df.sample(withReplacement = false, underSampleRatio, seed = 42L).collect()
      samples = samples ++ currentSamples
      val foo = spark.sparkContext.parallelize(samples)
      val x = spark.sqlContext.createDataFrame(foo, df.schema)
      x
    }
    else {
      df
    }
  }

  def smoteSample(randomInts: Random, currentClassZipped: Array[(Row, Int)], cls: Int): Row = {
    def r = randomInts.nextInt(currentClassZipped.length)

    val rand = Array(r, r, r, r, r)
    val sampled: Array[Row] = currentClassZipped.filter(x => rand.contains(x._2)).map(x => x._1) //FIXME - issues not taking duplicates
    //FIXME - can we dump the index column?
    val values: Array[Array[Double]] = sampled.map(x=>x(2).asInstanceOf[DenseVector].toArray)

    val ddd: Array[Double] = values.transpose.map(_.sum /values.length)

    val r2 = Row(0.toLong, cls, Vectors.dense(ddd))
    r2
    }

  def smote(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    val aggregatedCounts = df.groupBy("label").agg(count("label"))
    val randomInts: Random = new scala.util.Random(42L)
    // FIXME - make this more parallel
    val currentCount = df.count()
    val cls = aggregatedCounts.take(1)(0)(0).toString.toInt //FIXME

    val finaDF = if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      val currentClassZipped = df.collect().zipWithIndex
      val mappedResults = spark.sparkContext.parallelize(1 to samplesToAdd.toInt).map(x => smoteSample(randomInts, currentClassZipped, cls))
      val mappedDF = spark.sqlContext.createDataFrame(mappedResults, df.schema)
      val joinedDF = df.union(mappedDF)
      joinedDF
    }
    else {
      df
    }
    finaDF
  }

  def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame = {
    val numberOfClasses = df.select("label").distinct().count()
    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(numberOfClasses.toInt) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x => (x(0).toString, x(1).toString.toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    spark.createDataFrame(rdd)
  }

  def convertFeaturesToVector(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._
    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    df.withColumn("features", convertToVector($"features"))
  }

  def main(args: Array[String]) {

    val filename = "/home/ford/data/sampling_input.txt"
    val lines = Source.fromFile(filename).getLines.map(x=>x.split(":")(0)->x.split(":")(1)).toMap
    val input_file = lines("dataset").trim
    val classifier = lines("classifier").trim
    val samplingMethods = lines("sampling").split(",").map(x=>x.trim)
    val labelColumnName = lines("labelColumn").trim
    val enableDataScaling = if(lines("enableScaling").trim == "true") true else false
    val numSplits = lines("numCrossValidationSplits").trim.toInt
    val savePath = lines("savePath").trim

    println("input")
    for(x<-lines) {
      println(x)
    }
    println("Dataset: " + input_file)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    val df = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv(input_file).withColumnRenamed(labelColumnName, "label")

    import spark.implicits._
    import spark.implicits._

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })



    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    results.show()
    results.printSchema()

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")

    val datasetIndexed = indexer.fit(results).transform(results).drop("label").withColumnRenamed("labelIndexed", "label")
      //.withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label")
      //.withColumnRenamed("labelIndexed",  "label")
    println("here")
    datasetIndexed.show()
    datasetIndexed.printSchema()


    //val labelMap = datasetIndexed.select("originalLabel", "label").distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    //val labelMapReversed = labelMap.map(x=>(x._2, x._1))



    val converted: DataFrame = convertFeaturesToVector(datasetIndexed)

    /*

    val asDense = udf((v: SparseVector) => {
      val denseVector = v.toDense
      org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
    })// v.toDense)

     */

    // val asDense = udf((v: SparseVector) => v.toDense)

    val asDense = udf((v: Any) => {
      if(v.isInstanceOf[SparseVector]) {
        v.asInstanceOf[SparseVector].toDense
      } else {
        v.asInstanceOf[DenseVector]
      }
    })

    val scaledData1: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      // scaledData.show(1000)
      scaledData.printSchema()
      scaledData.drop("features").withColumn("features", asDense($"scaledFeatures")).drop("scaledFeatures")
      //scaledData.drop("features").withColumn("features", $"scaledFeatures").drop("scaledFeatures")

  //.withColumnRenamed("scaledFeatures", "features") ..
    } else { converted }.cache()

    val scaledData = scaledData1//.filter(scaledData1("label")===1.0 || scaledData1("label")===2.0 || scaledData1("label")===3.0)

    // FIXME - add pipeline

    val counts = scaledData.count()
    var splits = Array[Int]()
    println("scaled data")
    scaledData.show()
    scaledData.printSchema()

    splits :+= 0

    if(numSplits < 2) {
      splits :+= (counts * 0.2).toInt
    }
    else {
      for(i <- 1 until numSplits) {
        splits :+= ((counts / numSplits) * i).toInt
      }
    }
    splits :+= counts.toInt

    var combinedSplitResults =  Array[Array[String]]()

    for(splitIndex<-0 until numSplits) {
      val testData = scaledData.filter(scaledData("index") < splits(splitIndex + 1) && scaledData("index") >= splits(splitIndex)).persist()
      val trainData = scaledData.filter(scaledData("index") >= splits(splitIndex + 1) || scaledData("index") < splits(splitIndex)).persist()

      getCountsByClass(spark, "label", trainData).show

      println("original size: " + trainData.count())

      val samplingMap: Map[String, Double] =
        Map( "1" -> 2.0,
          "2" -> 0.5,
          "3" -> 2.0,
          "4" -> 1.0,
          "5" -> 2.0,
          "6" -> 2.0,
          "7" -> 2.0

        /*Map( "none" -> 1.0,
          "rRna" -> 10.0,
          "tRna" -> 20.0,
          "snRna" -> 20.0,
          "mRna" -> 40.0,
          "SRP" -> 40.0,
          "IRES" -> 40.0*/
        )

    // FIXME - make per class map parallel to run faster

     var resultArray = Array[Array[String]]()
     for(samplingMethod <- samplingMethods) {
       println("$$$$$$$$$$$$$$$$$$$" + samplingMethod + "$$$$$$$$$$$$$$$$$$$")
       val t0 = System.nanoTime()
       val sampledData = if(samplingMethod == "kMeansSmote") {
         val r = new KMeansSMOTE()
         val model = r.fit(trainData).setBalanceThreshold(0.0)//.setTopTreeSize(2)
         model.transform(trainData)
       }  else if(samplingMethod == "borderlineSmote") {
         val r = new BorderlineSMOTE()
         val model = r.fit(trainData).setBalanceThreshold(0.0)
         model.transform(trainData)
       }  else if(samplingMethod == "rbo") {
         val r = new RBO()
         val model = r.fit(trainData)
         model.transform(trainData)
       } else if(samplingMethod == "adasyn") {
         val r = new ADASYN().setBalanceThreshold(0.0)
         val model = r.fit(trainData)
         model.transform(trainData)
       } else if(samplingMethod == "safeLevel") {
         val r = new SafeLevelSMOTE()
         val model = r.fit(trainData)
         model.transform(trainData)
       } else if(samplingMethod == "smote") {
         val r = new SMOTE
         val model = r.fit(trainData).setBalanceThreshold(0.0).setTopTreeSize(2)   //.setSamplingRatios(samplingMap)
         model.transform(trainData)
       } else if(samplingMethod == "mwmote") {
         val r = new MWMOTE()
         val model = r.fit(trainData).setBalanceThreshold(0.0).setTopTreeSize(2)
         model.transform(trainData)
       } else if(samplingMethod == "ccr") {
         val r = new CCR()
         val model = r.fit(trainData).setBalanceThreshold(0.0)// .setTopTreeSize(10)
         model.transform(trainData)
       } else if(samplingMethod == "ans") {
         val r = new ANS()
         val model = r.fit(trainData).setBalanceThreshold(0.0)
         model.transform(trainData)
       } else if(samplingMethod == "clusterSmote") {
         val r = new ClusterSMOTE()
         val model = r.fit(trainData).setBalanceThreshold(0.0).setTopTreeSize(2)
         model.transform(trainData)
       } else if(samplingMethod == "gaussianSmote") {
         val r = new GaussianSMOTE()
         val model = r.fit(trainData).setBalanceThreshold(0.0)
         model.transform(trainData)
       } else if(samplingMethod == "smote_d") {
         val r = new SMOTED()
         val model = r.fit(trainData).setBalanceThreshold(0.0)
         model.transform(trainData)
       } else if(samplingMethod == "nras") {
         val r = new NRAS()
         val model = r.fit(trainData)
         model.transform(trainData)
       }  else if(samplingMethod == "randomOversample") {
         val r = new RandomOversample() // multi-class done
         val model = r.fit(trainData)
         model.transform(trainData)
       }  else if(samplingMethod == "randomUndersample") {
         val r = new RandomUndersample() // multi-class done
         val model = r.fit(trainData)
         model.transform(trainData)
       }
       else {
         //sampleData(spark, trainData, samplingMethod)
         trainData
       }
       // sampledData.show

       println("new total count: " + sampledData.count())
       getCountsByClass(spark, "label", sampledData).show
       sampledData.printSchema()

       val t1 = System.nanoTime()

       val savePathString = savePath
       val saveDirectory = new File(savePathString)
       if (!saveDirectory.exists()) {
         saveDirectory.mkdirs()
       }

       val x: Array[String] = Array(samplingMethod) ++ runClassifierMinorityType(sampledData, testData) ++ Array(((t1 - t0) / 1e9).toString)

       resultArray = resultArray :+ x
       combinedSplitResults = combinedSplitResults :+ x
     }

     val resultsDF = buildResultDF(spark, resultArray)
     println("Split Number: " + splitIndex)
     resultsDF.show

     resultsDF.repartition(1).
       write.format("com.databricks.spark.csv").
       option("header", "true").
       mode("overwrite").
       save(savePath + "/" + splitIndex)

     trainData.unpersist()
     testData.unpersist()
    }

    // println("Total")
    val totalResults = buildResultDF(spark, combinedSplitResults)
    val totals = totalResults.groupBy("sampling").agg(avg("AvAvg").as("AvAvg"),
      avg("MAvG").as("MAvG"), avg("RecM").as("RecM"), avg("Recu").as("Recu"),
      avg("PrecM").as("PrecM"), avg("Precu").as("Precu"), avg("FbM").as("FbM"),
      avg("Fbu").as("Fbu"), avg("AvFb").as("AvFb"), avg("CBA").as("CBA"), avg("time").as("time"))
    totals.show

    totals.repartition(1).
      write.format("com.databricks.spark.csv").
      option("header", "true").
      mode("overwrite").
      save(savePath + "/totals")
  }

  def buildResultDF(spark: SparkSession, resultArray: Array[Array[String]] ): DataFrame = {
    import spark.implicits._

    for(x<-resultArray.indices) {
      for(y<-resultArray(x).indices) {
        // println(y)
      }
    }
    val csvResults = resultArray.map(x => x match {
      case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) => (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
    }).toSeq
    val c = spark.sparkContext.parallelize(csvResults).toDF
    val lookup = Map(
      "_1" -> "sampling",
      "_2" -> "AvAvg",
      "_3" -> "MAvG",
      "_4" -> "RecM",
      "_5" -> "Recu",
      "_6" -> "PrecM",
      "_7" -> "Precu",
      "_8" -> "FbM",
      "_9" -> "Fbu",
      "_10" -> "AvFb",
      "_11" -> "CBA",
      "_12" -> "time"
    )

    val cols = c.columns.map(name => lookup.get(name) match {
      case Some(newName) => col(name).as(newName)
      case None => col(name)
    })

    c.select(cols: _*)
  }

  def runClassifierMinorityType(train: DataFrame, test: DataFrame): Array[String] ={

    import train.sparkSession.implicits._
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")


    val indexerModel = indexer.fit(train)
    val indexedTrain = indexerModel.transform(train).withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label")
    val indexedTest = indexerModel.transform(test).withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label")//.sample(0.05)

    val labelMap = indexedTrain.select("originalLabel", "label").distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    val labelMapReversed = labelMap.map(x=>(x._2, x._1))

    println("^^ runClassifierMinorityType")
    println(labelMapReversed)
    //val spark = train.sparkSession
    //FIXME - don't collect twice
    //val maxLabel: Double = indexedTest.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble).max
    //val minLabel: Double = indexedTest.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble).min
    // val inputCols = test.columns.filter(_ != "label")

    val classifier = new RandomForestClassifier().setNumTrees(10).
      setSeed(42L).
      setLabelCol("label").
      setFeaturesCol("features").
      setPredictionCol("prediction")

    indexedTrain.show
    indexedTrain.printSchema()


    val model = classifier.fit(indexedTrain)
    val predictions: DataFrame = model.transform(indexedTest)

    predictions.show(100)
    predictions.printSchema()
    //predictions.select("label", "prediction").write.format("csv").option("header", "false").save("/home/ford/tmp/predictions/")

    import org.apache.spark.ml.linalg.Vectors

    //val predictionAndLabels = test.select("label", "prediction").collect().map(x=>(x(0).asInstanceOf[Double], x(1).asInstanceOf[DenseVector])).map { case LabeledPoint(label, features) =>
    /*val predictionAndLabels = test.select("label", "features").map(row =>
      val prediction = model.predict(features)
      //new LabeledPoint(row(0).asInstanceOf[Double],  org.apache.spark.mllib.linalg.Vectors.fromML(row(1).asInstanceOf[DenseVector]))).rdd
      new LabeledPoint(row(0).asInstanceOf[Double],prediction)).rdd
*/

    import org.apache.spark.mllib.linalg.Vectors
    val predictionAndLabels = test.select("label", "features").map { row =>
      val f = row(1).asInstanceOf[DenseVector]

      val prediction = model.predict(f)
      (prediction, row(0).asInstanceOf[Double])
    }.rdd


    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion matrix:")
    println(metrics.confusionMatrix)
    println(metrics)

    val labels: Array[Double] = labelMap.values.toArray.sorted

    val confusionMatrix: Dataset[Row] = predictions.groupBy("label").
      pivot("prediction", labels).
      count().
      na.fill(0.0).
      orderBy("label")

    println("cm")
    confusionMatrix.printSchema()

    calculateClassifierResults(indexedTest.select("label").distinct(), confusionMatrix, labels)
    //Array[String]()
  }




  /*************************************************/



  def sampleData(spark: SparkSession, df: DataFrame, samplingMethod: String): DataFrame = {
    val d = df.select("label").distinct()
    val presentClasses: Array[Int] = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt)

    val counts = getCountsByClass(spark, "label", df)
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    val minClassCount = counts.select("_2").agg(min("_2")).take(1)(0)(0).toString.toInt

    val overSampleCount = maxClassCount
    val underSampleCount = minClassCount
    val smoteSampleCount = maxClassCount // / 2

    val myDFs: Array[(Int, DataFrame)] = presentClasses.map(x=>(x, df.filter(df("label") === x).toDF()))

    val classDF: Array[DataFrame] = presentClasses.map(x => sampleDataParallel(spark, myDFs.filter(y=>y._1 == x)(0)._2, x, samplingMethod, underSampleCount, overSampleCount, smoteSampleCount))
    val r = classDF.reduce(_ union _)
    r
  }


  def sampleDataParallel(spark: SparkSession, df: DataFrame, presentClass: Int, samplingMethod: String, underSampleCount: Int, overSampleCount: Int, smoteSampleCount: Int): DataFrame = {
    val l = presentClass
    val currentCase = df.filter(df("label") === l).toDF()
    val filteredDF2 = samplingMethod match {
      case "undersample" => underSample(spark, currentCase, underSampleCount)
      case "oversample" => overSample(spark, currentCase, overSampleCount)
      case "smote" => smote(spark, currentCase, smoteSampleCount)
      case _ => currentCase
    }
    filteredDF2
  }

}
