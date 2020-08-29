package org.apache.spark.ml.sampling

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.io.Source
import scala.util.Random

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

  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame): Array[String]={//String ={
    import distinctClasses.sparkSession.implicits._
    //FIXME - don't calculate twice

    val classLabels = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt)

    val maxLabel: Int = classLabels.max
    val minLabel: Int = classLabels.min
    val numberOfClasses = classLabels.length
    val classCount = confusionMatrix.columns.length - 1
    val testLabels = distinctClasses.map(_.getAs[Int]("label")).map(x => x.toInt).collect().sorted

    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum
    val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))

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

    //FIXME - could be made parallel w/udf
    for (clsIndex <- minLabel to maxLabel - minLabel) {
      val colSum = rows.map(x => x(clsIndex + 1).toInt).sum
      val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x.head.toDouble.toInt == clsIndex)(0).tail.map(x => x.toDouble.toInt).sum else 0
      val tp: Double = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x.head.toDouble.toInt == clsIndex)(0).tail(clsIndex).toDouble.toInt else 0
      val fn: Double = colSum - tp
      val fp: Double = rowValueSum - tp
      val tn: Double = totalCount - tp - fp - fn

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

    AvAvg /= classCount
    MAvG = {  val result = Math.pow(MAvG, 1/numberOfClasses.toDouble); if(result.isNaN) 0.0 else result } //Math.pow((MAvG), (1/numberOfClasses.toDouble))
    RecM /= classCount
    PrecM /= classCount
    Recu = tpSum / tSum
    Precu = tpSum / pSum
    FbM = { val result = ((1 + Math.pow(beta, 2.0)) * PrecM * RecM) / (Math.pow(beta, 2.0) * PrecM + RecM); if(result.isNaN) 0.0 else result }
    Fbu = { val result = ((1 + Math.pow(beta, 2.0)) * Precu * Recu) / (Math.pow(beta, 2.0) * Precu + Recu); if(result.isNaN) 0.0 else result }
    AvFb /= classCount
    CBA /= classCount

    Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)
  }

  //assume there is only one class present
  def overSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]() //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
    //val samplesToAdd = numSamples - df.count()
    val currentCount = df.count()
    if (0 < currentCount && currentCount < numSamples) {
      val currentSamples = df.sample(true, (numSamples - currentCount) / currentCount.toDouble).collect()
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
      val currentSamples = df.sample(false, underSampleRatio, seed = 42L).collect()
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
    val cls = aggregatedCounts.take(1)(0)(0).toString().toInt //FIXME

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

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    val df = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv(input_file).withColumnRenamed(labelColumnName, "label")

    import spark.implicits._

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, Int, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString.toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    val converted: DataFrame = convertFeaturesToVector(results)

    val scaledData: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.drop("features").withColumnRenamed("scaledFeatures", "features")
    } else { converted }.cache()

    val counts = scaledData.count()
    var splits = Array[Int]()
    scaledData.show()


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

      // Add: random oversampling, random undersampling

     /* val smote = new ClassBalancerModel()
      val result = smote.transform(trainData)

      println("******* After train ")
      testData.show
      println(testData.count)
      result.show
      println(result.count)

      getCountsByClass(spark, "label", result).show
         }*/

           var resultArray = Array[Array[String]]()
           for(samplingMethod <- samplingMethods) {
             println(samplingMethod)
             val t0 = System.nanoTime()
             val sampledData = if(samplingMethod == "kMeansSmote") {
               val r = new KMeansSMOTE()
               val model = r.fit(trainData)
               model.transform(trainData)
             }  else if(samplingMethod == "borderlineSmote") {
               val r = new BorderlineSMOTE()
               val model = r.fit(trainData)
               model.transform(trainData)
             }  else if(samplingMethod == "rbo") {
               val r = new RBO()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "adasyn") {
               val r = new ADASYN()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "safeLevel") {
               val r = new SafeLevelSMOTE()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "SMOTE") {
               val r = new SMOTE
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "mwmote") {
               val r = new MWMOTE()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "ccr") {
               val r = new CCR()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "ans") {
               val r = new ANS()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "clusterSmote") {
               val r = new ClusterSMOTE()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "gaussianSmote") {
               val r = new GaussianSMOTE()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "smote_d") {
               val r = new SMOTED()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "somo") {
               val r = new SOMO()
               val model = r.fit(trainData)
               model.transform(trainData)
             } else if(samplingMethod == "nras") {
               val r = new NRAS()
               val model = r.fit(trainData)
               model.transform(trainData)
             }
             else {
               sampleData(spark, trainData, samplingMethod)
             }
             // sampledData.show

             println("new total count: " + sampledData.count())

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

    println("Total")
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
    val spark = train.sparkSession
    //FIXME - don't collect twice
    val maxLabel: Int = test.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble.toInt).max
    val minLabel: Int = test.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble.toInt).min
    val inputCols = test.columns.filter(_ != "label")

    val classifier = new RandomForestClassifier().setNumTrees(10).
      setSeed(42L).
      setLabelCol("label").
      setFeaturesCol("features").
      setPredictionCol("prediction")

    val model = classifier.fit(train)
    val predictions = model.transform(test)

    val confusionMatrix: Dataset[Row] = predictions.
      groupBy("label").
      pivot("prediction", minLabel to maxLabel).
      count().
      na.fill(0.0).
      orderBy("label")

    calculateClassifierResults(test.select("label").distinct(), confusionMatrix)
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
