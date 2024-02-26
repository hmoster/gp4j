package uk.ac.soton.ecs.gp4j.example;

import java.util.Map;
import java.util.Random;

import Jama.Matrix;
import org.apache.commons.lang.ArrayUtils;

import uk.ac.soton.ecs.gp4j.bmc.BasicPrior;
import uk.ac.soton.ecs.gp4j.bmc.GaussianProcessMixture;
import uk.ac.soton.ecs.gp4j.bmc.GaussianProcessRegressionBMC;
import uk.ac.soton.ecs.gp4j.gp.GaussianProcess;
import uk.ac.soton.ecs.gp4j.gp.GaussianProcessPrediction;
import uk.ac.soton.ecs.gp4j.gp.GaussianProcessRegression;
import uk.ac.soton.ecs.gp4j.gp.covariancefunctions.*;
import uk.ac.soton.ecs.gp4j.util.MatrixUtils;

import static jdk.nashorn.internal.objects.Global.println;

public class GuassianProcessRegressionExampleOriginal {

	public static void main(String[] args) {
		// normalGPExample();

		learningGPExample2D();

	}

	private static double objective(double x) {
		return -(Math.pow(x-3, 2) + 2);
	}

	private static double objective2D(double x1, double x2) {
		return -Math.pow(x1, 2) - Math.pow((x2 - 1), 2) + 1;
	}

	private static double objective3D(double x1, double x2, double x3) {
		return -Math.pow(x1, 2) - Math.pow((x2 - 1), 2) * x3 + 1;
	}

	private static void learningGPExample() {
		GaussianProcessRegressionBMC regression = new GaussianProcessRegressionBMC();
		//regression.setCovarianceFunction(new SumCovarianceFunction(
		//		SquaredExponentialCovarianceFunction.getInstance(),
		//		NoiseCovarianceFunction.getInstance()));
		regression.setCovarianceFunction(new SumCovarianceFunction(
				        new Matern3ARDCovarianceFunction(),
						NoiseCovarianceFunction.getInstance()));

		//regression.setPriors(new BasicPrior(11, 1.0, 2.0), new BasicPrior(11,
		//		1.0, 2.0), new BasicPrior(1, .01, 1.0));
		regression.setPriors(new BasicPrior(1, 1.0, 1.0), new BasicPrior(1,
				1.0, 1.0),new BasicPrior(1, 1.0, 1.0));

		//double[] trainX = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
		//double[] trainY = new double[] { 1.0, 1.0, 1.0, 1.5, .5, 0.0 };
		double[] trainX = { -1.6595599059485195, 4.4064898688431615, -9.997712503653101, -3.9533485473632046 };
		//double[] trainY = { -23.711498517122976, -3.978213751158454, -170.94053032762017, -50.34905602111799 };
		//double[] trainY = { 0.59394467, 0.89810942, -1.675413, 0.1833589 };
		double[] trainY = { -23.711498517122976, -3.978213751158454, -170.94053032762017, -50.34905602111799 };

		/*
		Random rand = new Random(12345);
		double[] trainX = new double[5];
		double[] trainY = new double[5];
		for(int i=0; i<5; i++) {
			double p = rand.nextDouble() * (20 - (-10)) + (-10);
			trainX[i] = p;
			trainY[i] = objective(p);
        }
        */
		GaussianProcessMixture predictor = regression.calculateRegression(
				trainX, trainY);

		Map<Double[], Double> hyperParameterWeights = regression
				.getHyperParameterWeights();

		for (Double[] hyperparameterValue : hyperParameterWeights.keySet()) {
			System.out.println("hyperparameter value: "
					+ ArrayUtils.toString(hyperparameterValue) + ", weight: "
					+ hyperParameterWeights.get(hyperparameterValue));
		}

		double[] testX = new double[] { -1.6595599059485195, -5.0, -4.0};

		//double[] testX = new double[] { -1.6595599059485195 };
		GaussianProcessPrediction prediction = predictor
				.calculatePrediction(MatrixUtils.createColumnVector(testX));

		prediction.getMean().print(1, 17);
		prediction.getVariance().print(1, 17);
		prediction.getStandardDeviation().print(1, 17);

	}

	private static void learningGPExample2D() {
		GaussianProcessRegressionBMC regression = new GaussianProcessRegressionBMC();
		regression.setCovarianceFunction(new SumCovarianceFunction(
				new Matern3ARDCovarianceFunction(),
				NoiseCovarianceFunction.getInstance()));

		//regression.setPriors(new BasicPrior(11, 1.0, 2.0), new BasicPrior(11,
		//		1.0, 2.0), new BasicPrior(1, .01, 1.0));
		//regression.setPriors(new BasicPrior(1, 1.0, 1.0), new BasicPrior(1,
		//		1.0, 1.0),new BasicPrior(1, 1.0, 1.0));

		regression.setPriors(new BasicPrior(11, 1.0, 2.0),
				new BasicPrior(11,1.0, 2.0),
				new BasicPrior(11, 1.0, 2.0),
				new BasicPrior(1, .01, 1.0));
		//double[] trainX = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
		//double[] trainY = new double[] { 1.0, 1.0, 1.0, 1.5, .5, 0.0 };
		//double[] trainX = { -1.6595599059485195, 4.4064898688431615, -9.997712503653101, -3.9533485473632046 };
		//double[] trainY = { -23.711498517122976, -3.978213751158454, -170.94053032762017, -50.34905602111799 };

		Random rand = new Random(12345);
		double[][] trainX = new double[20][2];
		double[][] trainY = new double[20][1];
		for(int i=0; i<20; i++) {
			double x1 = rand.nextDouble() * (20 - (-10)) + (-10);
			double x2 = rand.nextDouble() * (20 - (-10)) + (-10);
			trainX[i][0] = x1;
			trainX[i][1] = x2;
			trainY[i][0] = objective2D(x1, x2);
		}

		GaussianProcessMixture predictor = regression.calculateRegression(
				new Matrix(trainX), new Matrix(trainY));

		Map<Double[], Double> hyperParameterWeights = regression
				.getHyperParameterWeights();

		for (Double[] hyperparameterValue : hyperParameterWeights.keySet()) {
			System.out.println("hyperparameter value: "
					+ ArrayUtils.toString(hyperparameterValue) + ", weight: "
					+ hyperParameterWeights.get(hyperparameterValue));
		}

		//double[] testX = new double[] { -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
		//		8.0, 9.0, 10.0 };

		Matrix testX = new Matrix(new double[][] {{1.0, 7.0}});
		GaussianProcessPrediction prediction = predictor
				.calculatePrediction(testX);

		prediction.getMean().print(1, 3);
		prediction.getVariance().print(1, 3);
		prediction.getStandardDeviation().print(1, 3);
	}

	private static void learningGPExample3D() {
		GaussianProcessRegressionBMC regression = new GaussianProcessRegressionBMC();
		regression.setCovarianceFunction(new SumCovarianceFunction(
				new Matern3ARDCovarianceFunction(),
				NoiseCovarianceFunction.getInstance()));

		regression.setPriors(new BasicPrior(11, 1.0, 2.0), new BasicPrior(11,
				1.0, 2.0), new BasicPrior(1, .01, 1.0));

		//double[] trainX = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
		//double[] trainY = new double[] { 1.0, 1.0, 1.0, 1.5, .5, 0.0 };
		//double[] trainX = { -1.6595599059485195, 4.4064898688431615, -9.997712503653101, -3.9533485473632046 };
		//double[] trainY = { -23.711498517122976, -3.978213751158454, -170.94053032762017, -50.34905602111799 };

		double scalar = 5.0;

		// 将标量值转换成一个2x2的多维数组
		double[][] array = {{scalar, scalar}, {scalar, scalar}};

		// 创建一个Jama矩阵对象
		Matrix matrix = new Matrix(array);


		Random rand = new Random(12345);
		double[][] trainX = new double[20][3];
		double[][] trainY = new double[20][1];
		for(int i=0; i<20; i++) {
			double x1 = rand.nextDouble() * (20 - (-10)) + (-10);
			double x2 = rand.nextDouble() * (20 - (-10)) + (-10);
			double x3 = rand.nextDouble() * (20 - (-10)) + (-10);
			trainX[i][0] = x1;
			trainX[i][1] = x2;
			trainX[i][2] = x3;
			trainY[i][0] = objective3D(x1, x2, x3);
		}

		GaussianProcessMixture predictor = regression.calculateRegression(
				new Matrix(trainX), new Matrix(trainY));

		Map<Double[], Double> hyperParameterWeights = regression
				.getHyperParameterWeights();

		for (Double[] hyperparameterValue : hyperParameterWeights.keySet()) {
			System.out.println("hyperparameter value: "
					+ ArrayUtils.toString(hyperparameterValue) + ", weight: "
					+ hyperParameterWeights.get(hyperparameterValue));
		}

		//double[] testX = new double[] { -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
		//		8.0, 9.0, 10.0 };

		Matrix testX = new Matrix(new double[][] {{2.0, 5.0, 3.0}, {4.0, 2.0, 6.0}});
		GaussianProcessPrediction prediction = predictor
				.calculatePrediction(testX);

		prediction.getMean().print(1, 1);
		prediction.getVariance().print(1, 1);

	}

	/*
	private static void normalGPExample() {
		GaussianProcessRegression regression = new GaussianProcessRegression(
				new double[] { 0.0, 0.0 }, SquaredExponentialCovarianceFunction
						.getInstance());

		double[] trainX = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
		double[] trainY = new double[] { 1.0, 4.0, 3.0, 7.0, 6.0, 5.0 };

		GaussianProcess predictor = regression.calculateRegression(trainX,
				trainY);

		double[] testX = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
				8.0, 9.0, 10.0 };

		GaussianProcessPrediction prediction = predictor
				.calculatePrediction(MatrixUtils.createColumnVector(testX));

		prediction.getMean().print(10, 10);
		prediction.getVariance().print(10, 10);
	}
	*/
}
