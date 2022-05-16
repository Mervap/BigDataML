package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector, norm, sum}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, linalg}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol {
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val epochs = new IntParam(this, "epochs", "Count of epochs to fit")
  def getEpochs: Int = $(epochs)
  def setEpochs(value: Int): this.type = set(epochs, value)

  val learningRate = new DoubleParam(this, "learningRate", "Learning rate of each step")
  def getLearningRate: Double = $(learningRate)
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  val tolerance = new DoubleParam(this, "tolerance", "Tolerance of solution")
  def getTolerance: Double = $(tolerance)
  def setTolerance(value: Double): this.type = set(tolerance, value)

  val batchSize = new IntParam(this, "batchSize", "Batch of gradient descent size")
  def getBatchSize: Int = $(batchSize)
  def setBatchSize(value: Int): this.type = set(batchSize, value)

  setDefault(epochs -> 200, learningRate -> 1.0, tolerance -> 1e-6, batchSize -> 10000, inputCol -> "features")

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val assembler = new VectorAssembler().setInputCols(Array(getInputCol, getLabelCol)).setOutputCol("tmp")
    val vectors = assembler.transform(dataset).select("tmp").as[Vector]

    val dim = vectors.first().size - 1

    val epochs = getEpochs
    val lr = getLearningRate
    val eps = getTolerance
    val batchSize = getBatchSize
    var weight = DenseMatrix.zeros[Double](dim, 1)
    var diffNorm = Double.PositiveInfinity

    var curEpochs = 0
    while (curEpochs < epochs && diffNorm > eps) {
      val grad = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        Iterator(
          data.sliding(batchSize, batchSize).foldLeft(new MultivariateOnlineSummarizer())(
            (summarizer, vectors) => {
              val rows = vectors.map(row => row.asBreeze(0 until dim).toArray).toArray
              val matrix = new DenseMatrix(rows.apply(0).length, vectors.size, rows.flatten, 0).t
              val y = new DenseVector(vectors.map(x => x.asBreeze(-1)).toArray).asDenseMatrix.t

              val grads = matrix.t * (matrix * weight - y) / vectors.length.toDouble
              summarizer.add(mllib.linalg.Vectors.fromBreeze(grads.toDenseVector))
            }
          )
        )
      }).reduce(_ merge _)

      val diff = lr * grad.mean.asBreeze
      weight = DenseMatrix.create(weight.rows, weight.cols, (weight.toDenseVector - diff).toArray)
      curEpochs += 1
      diffNorm = norm(diff)
    }

    copyValues(new LinearRegressionModel(weight.flatten())).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
  override val uid: String,
  val weight: DenseVector[Double],
) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this(weight: DenseVector[Double]) = this(Identifiable.randomUID("LinearRegressionModel"), weight)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weight), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = {
      dataset.sqlContext.udf.register(
        uid + "_transform",
        (x: linalg.DenseVector) => sum(x.asBreeze.toDenseVector * weight)
      )
    }
    dataset.withColumn(getOutputCol, transformUdf(dataset(getInputCol)))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val tupled = Tuple1(Vectors.fromBreeze(weight))
      sqlContext.createDataFrame(Seq(tupled)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val weight = vectors.select(vectors("_1").as[Vector]).first().asBreeze.toDenseVector
      val model = new LinearRegressionModel(weight)
      metadata.getAndSetParams(model)
      model
    }
  }
}
