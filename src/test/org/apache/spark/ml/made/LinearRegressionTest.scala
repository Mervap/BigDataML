package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val eps = 0.0001
  lazy val testData: Dataset[_] = LinearRegressionTest._testData.withColumn("y", toLabelColumn(col("features")))
  lazy val labels: DenseVector[Double] = LinearRegressionTest._labels
  lazy val weight: DenseVector[Double] = DenseVector(LinearRegressionTest._weight.toArray)
  lazy val toLabelColumn: UserDefinedFunction = LinearRegressionTest._toLabelColumn

  "Model" should "predict function" in {
    val model = new LinearRegressionModel(weight).setOutputCol("pred")
    validateModel(model.transform(testData))
  }

  "Estimator" should "calculate correct weights" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setLabelCol("y")
      .setOutputCol("pred")

    val model = estimator.fit(testData)
    validateWeight(model.weight)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setLabelCol("y")
        .setOutputCol("pred")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(testData).stages(0).asInstanceOf[LinearRegressionModel]

    validateWeight(model.weight)
    validateModel(model.transform(testData))
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setLabelCol("y")
        .setOutputCol("pred")
    ))
    val model = pipeline.fit(testData)

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.transform(testData))
  }

  private def validateModel(actual: DataFrame): Unit = {
    val pred = actual.select("pred").collect()
    pred.length should be(labels.length)
    for (i <- pred.indices) {
      pred.apply(i).getDouble(0) should be(labels(i) +- eps)
    }
  }

  private def validateWeight(actual: DenseVector[Double]): Unit = {
    for (i <- 0 until weight.length) {
      actual(i) should be(weight(i) +- eps)
    }
  }
}

object LinearRegressionTest extends WithSpark {

  import sqlc.implicits._

  lazy val _weight: Vector = Vectors.dense(1.5, 0.3, -0.7)

  lazy val _toLabelColumn: UserDefinedFunction = udf { x: Vector =>
    (x.asBreeze.toDenseVector.asDenseMatrix * DenseVector(_weight.toArray)).data.apply(0)
  }

  private lazy val _features: Matrix = Matrices.dense(100000, 3, DenseMatrix.rand[Double](100000, 3).toArray)
  lazy val _labels: DenseVector[Double] = DenseVector(_features.multiply(_weight).toArray)

  lazy val _testData: DataFrame = {
    val matrixRows = _features.rowIter.toSeq.map(_.toArray)
    spark.sparkContext.parallelize(matrixRows.map(x => Tuple1(Vectors.dense(x)))).toDF("features")
  }
}
