package uk.ac.soton.ecs.gp4j.gp

import Jama.{CholeskyDecomposition, Matrix}
import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, LBFGSB}
import uk.ac.soton.ecs.gp4j.gp.covariancefunctions.{CovarianceFunction, Matern3ARDCovarianceFunction}

import scala.util.Random

object GetBestLengthScale {
  //var initLoghyper: Array[Double] = Array()

  def costFunc(covarianceFunction: CovarianceFunction, lengthScale: Array[Double], trainX: Matrix, trainY: Matrix): (Double, Matrix, Matrix) = {
    val K = covarianceFunction.calculateCovarianceMatrix(lengthScale, trainX)

    for (i <- 0 until K.getRowDimension) {
      K.set(i, i, K.get(i, i) + 1e-6)
    }

    val chol = K.chol
    val L = chol.getL
    val alpha = chol.solve(trainY)
    val logLikelihood = new LogLikelihood(trainY, alpha, L)
    (logLikelihood.getValue, K, alpha)
  }

  def evalGradient(covarianceFunction: CovarianceFunction, lengthScale: Array[Double], trainX: Matrix,
                   K: Matrix, alpha: Matrix): DenseVector[Double] = {
    var inner_term = alpha.times(alpha.transpose())
    val identityMatrix: Matrix = Matrix.identity(K.getRowDimension, K.getColumnDimension)
    val chol = K.chol
    val K_inv = chol.solve(identityMatrix)

    inner_term = inner_term.minus(K_inv)

    val K_gradient = covarianceFunction.calculateTrainGradientMatrix(lengthScale, trainX)

    var log_likelihood_gradient_dims = 0.0
    for (i <- 0 until inner_term.getRowDimension) {
      for (j <- 0 until K_gradient.getColumnDimension) {
        log_likelihood_gradient_dims += inner_term.get(i, j) * K_gradient.get(j, i)
      }
    }
    //val log_likelihood_gradient =

    DenseVector(-0.5*log_likelihood_gradient_dims)
  }

  def chooseHyperparams(covarianceFunction: CovarianceFunction, trainX: Matrix, trainY: Matrix): Array[Double] = {
    //Choose hyperparameters based on maximizing the log - marginal
    //likelihood(potentially starting from several initial values)
    val minArray: Array[Double] = Array(math.log(1e-5))
    val maxArray: Array[Double] = Array(math.log(1e5))
    val solver = new LBFGSB(DenseVector[Double](minArray), DenseVector[Double](maxArray))

    val f = new DiffFunction[DenseVector[Double]] {
      override def calculate(lengthScale: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val input: Array[Double] = Array(lengthScale.valueAt(0), 0.0, 0.0)
        val cost = costFunc(covarianceFunction, input, trainX, trainY)
        val grad = evalGradient(covarianceFunction, input, trainX, cost._2, cost._3)
        (-cost._1, grad)
      }
    }

    var optX = solver.minimize(f, DenseVector[Double](0.0))
    var minX = optX.toArray
    var minY = f(optX)
    //Additional runs are performed from log -uniform chosen initial
    //theta
    val nIter = 5
    val rand = new Random(12345)
    val xIters = (0 until nIter).map(_ => rand.nextDouble() * (math.log(1e5) - math.log(1e-5)) + math.log(1e-5)).toArray
    //val xIters: Vector[Double] = Vector(-1.91063895, 5.07315894, -11.51029189, -4.55146072, -8.1337462)
    xIters.foreach(x => {
      optX = solver.minimize(f, DenseVector[Double](x))
      //Select result from run with minimal(negative) log -marginal
      //likelihood
      val optMax = f(optX)

      if (optMax < minY) {
        minY = optMax
        minX = optX.toArray
      }
    }
    )

    minX
  }
}