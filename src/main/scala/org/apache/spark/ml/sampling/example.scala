package org.apache.spark.ml.sampling

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.io.Source
import scala.util.Random
import org.apache.spark.ml.sampling.Utilities.{convertFeaturesToVector, getCountsByClass}

import org.apache.spark.mllib.linalg.{Vector, Vectors}

//FIXME - turn classes back to Ints instead of Doubles
object Example {




  def main(args: Array[String]) {

    val filename = "/home/ford/data/sampling_input_kurgan.txt"
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

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString//.toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    results.show()
    results.printSchema()

   val converted: DataFrame = convertFeaturesToVector(results)
   // val converted = results
    converted.show()
    converted.printSchema()



    /*def asDense = udf((v: SparseVector) =>
       v.toDense
     )*/

    val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)


    val scaledData: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.show()
      scaledData.printSchema()

     //scaledData.drop("features").withColumn("features", asDense($"scaledFeatures")).drop("scaledFeatures")  //.withColumnRenamed("scaledFeatures", "features") ..
      scaledData.drop("features").withColumn("features", toDense($"scaledFeatures")).drop("scaledFeatures")  //.withColumnRenamed("scaledFeatures", "features") ..
     // scaledData
    } else { converted }.cache()

    scaledData.show()
    scaledData.printSchema()


    val columnToString = udf((d: Double) => d.toString)



    getCountsByClass("label", scaledData).show

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")


    val indexed = indexer.fit(scaledData).transform(scaledData).withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label").withColumn("label", columnToString($"label"))
    indexed.show()
    indexed.printSchema()

    val samplingMap: Map[String, Double] =
      Map("0" -> 1.0,
        "1" -> 1.0,
        "2" -> 10.0,
        "3" -> 10.0,
        "4" -> 1.0,
        "5" -> 1.0,
        "6" -> 5.0,
        "7" -> 1.0,
        "none" -> 10.0,
        "tRna" -> 10.0,
        "rRna" -> 20.0,
        "snRna" -> 20.0,
        "IRES" -> 40.0,
        "SRP" -> 40.0,
        "mRna" -> 40.0
      )


    val labelMap = indexed.select("originalLabel", "label").distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    val labelMapReversed = labelMap.map(x=>(x._2, x._1))
    println("label map: " + labelMap)

    println("sampling map: " + samplingMap)

    var mappings = Map[Double, Double]()

    for(label <- labelMap) {
        if(samplingMap contains label._1) {
          println(label._1, label._2, samplingMap(label._1))
          mappings = mappings ++ Map(label._2 -> samplingMap(label._1))
      }
    }

    /*val r = new SMOTE
    val model = r.fit(indexed).setSamplingRatios(mappings)
    val transformedData = model.transform(indexed)

    //val r = new ADASYN()
    //val model = r.fit(indexed).setSamplingRatios(mappings)
    //val transformedData = model.transform(indexed)




    transformedData.show

    getCountsByClass("label", transformedData).show

    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    val convertFeatures = udf((features: DenseVector) => features.toArray.mkString(","))

    val restored = transformedData.withColumn("originalLabel", restoreLabel($"label")).drop("label")
      .withColumn("features2", convertFeatures($"features")).drop("features").withColumnRenamed("features2", "features")
      .withColumnRenamed("originalLabel", "label").drop("index").repartition(1) //.withColumnRenamed("label", "labelIndex")
    restored.show
    restored.printSchema()

    getCountsByClass("label", restored).show

    //restored.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").mode("overwrite").save(savePath + ".csv")


    restored.write.format("csv").option("header", "false").save("/home/ford/data/kurgan/Sep2020/adasyn")*/
  }


}


/*    val correctFeatures =  udf((features: String) => {
      val f = features.substring(1, features.length - 1)
      Vectors.dense(f.split(",").map(x=>x.toDouble)).toDense
    })


    val df = dfIn.withColumn("features2", correctFeatures(dfIn("features"))).drop("features").withColumnRenamed("features2", "features").drop("minorityType")
    df.show
    df.printSchema()*/
