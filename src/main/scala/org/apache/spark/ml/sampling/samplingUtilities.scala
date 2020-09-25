package org.apache.spark.ml.sampling

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{count, udf}
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction

import scala.collection.mutable


object utils {
  type Element = (Long, (Int, Array[Float]))
  //type Element2 = ()

 def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame ={
    val numberOfClasses = df.select("label").distinct().count()
    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(numberOfClasses.toInt) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x => (x(0).toString, x(1).toString.toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    spark.createDataFrame(rdd)
  }

  // FIXME - this is for cases with self neighbors
  val getMatchingClassCount: UserDefinedFunction = udf((array: mutable.WrappedArray[Int], minorityClassLabel: Int) => {
    def isMajorityNeighbor(x1: Int, x2: Int): Int = {
      if(x1 == x2) {
        1
      } else {
        0
      }
    }
    array.tail.map(x=>isMajorityNeighbor(minorityClassLabel, x)).sum
  })

  // set for different methods
  def pointDifference(x1: Array[Double], x2: Array[Double]): Double = {
    val combined = Array[Array[Double]](x1, x2)
    // val difference: Array[Double] = combined.transpose.map(x=>Math.abs(x(0)-x(1)))
    // difference.sum
    val difference: Array[Double] = combined.transpose.map(x=>Math.pow(x(1)-x(0), 2))
    Math.sqrt(difference.sum)
  }

}
