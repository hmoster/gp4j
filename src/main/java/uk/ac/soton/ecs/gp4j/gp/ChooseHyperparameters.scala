package uk.ac.soton.ecs.gp4j.gp

import Jama.{CholeskyDecomposition, Matrix}
import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, LBFGSB}
import uk.ac.soton.ecs.gp4j.gp.covariancefunctions.CovarianceFunction

object Utils {

  def costFunc(theta: Double, covarianceFunction: CovarianceFunction): Double = {
    val trainingCovarianceMatrix = covarianceFunction.calculateCovarianceMatrix(loghyper, trainX)

    val chol = trainingCovarianceMatrix.chol
    cholTrainingCovarianceMatrix = chol.getL
    alpha = chol.solve(trainY)
    logLikelihood = new LogLikelihood(trainY, alpha, cholTrainingCovarianceMatrix)
  }

  def evalGradient(): DenseVector[Double] = {

  }

  def chooseHyperparams() {
    //Choose hyperparameters based on maximizing the log - marginal
    //likelihood(potentially starting from several initial values)

    val solver = new LBFGSB(DenseVector[Double](minArray.toArray), DenseVector[Double](maxArray.toArray))

    val f = new DiffFunction[DenseVector[Double]] {
      override def calculate(theta: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val cost = costFunc(theta)
        val grad = evalGradient()
        (cost, grad)
      }
    }

    val optX = solver.minimize(f, DenseVector[Double](x))

    //Additional runs are performed from log -uniform chosen initial
    //theta

    //Select result from run with minimal(negative) log -marginal
    //likelihood
    val lml_values = list(map(itemgetter(1), optima))
    self.kernel_.theta = optima[np.argmin(lml_values)][0]
    //self.kernel_._check_bounds_params()
    //self.log_marginal_likelihood_value_ = -np.min(lml_values)
  }
}