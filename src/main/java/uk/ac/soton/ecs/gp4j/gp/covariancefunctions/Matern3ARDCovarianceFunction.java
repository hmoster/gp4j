package uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import Jama.Matrix;

public class Matern3ARDCovarianceFunction implements CovarianceFunction {

	public Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX) {
		return calculateTrainTestCovarianceMatrix(loghyper, trainX);
	}

	public Matrix calculateTestCovarianceMatrix(double[] loghyper, Matrix testX) {
		return new Matrix(testX.getRowDimension(), 1, Math
				.exp(2 * loghyper[loghyper.length - 1]));
	}

	public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
			Matrix trainX, Matrix testX) {
		int samplesTrain = trainX.getRowDimension();
		int samplesTest = testX.getRowDimension();

		if (samplesTrain == 0 || samplesTest == 0)
			return new Matrix(samplesTrain, samplesTest);

		//double signalVariance = Math.exp(2 * loghyper[loghyper.length - 1]);

		double[][] trainXVals = scaleValues(trainX, loghyper);
		double[][] testXVals = scaleValues(testX, loghyper);

		double[][] result = new double[samplesTrain][samplesTest];

		for (int i = 0; i < samplesTrain; i++) {
			for (int j = 0; j < samplesTest; j++) {
				double sq_sq_dist = Math.sqrt(5 * calculateSquareDistance(
						trainXVals[i], testXVals[j]));

				result[i][j] = Math.exp(-sq_sq_dist)
						* (1 + sq_sq_dist + Math.pow(sq_sq_dist, 2)/3);
			}
		}

		return new Matrix(result);
	}


	public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper, Matrix trainX) {
		Matrix copyTrainX = trainX;
		int samplesTrain = trainX.getRowDimension();
		int samplesTest = copyTrainX.getRowDimension();

		if (samplesTrain == 0 || samplesTest == 0)
			return new Matrix(samplesTrain, samplesTest);

		//double signalVariance = Math.exp(2 * loghyper[loghyper.length - 1]);

		double[][] trainXVals = scaleValues(trainX, loghyper);
		double[][] testXVals = scaleValues(copyTrainX, loghyper);

		double[][] result = new double[samplesTrain][samplesTest];

		for (int i = 0; i < samplesTrain; i++) {
			for (int j = 0; j < samplesTest; j++) {
				double sq_sq_dist = Math.sqrt(5 * calculateSquareDistance(
						trainXVals[i], testXVals[j]));

				result[i][j] = Math.exp(-sq_sq_dist)
						* (1 + sq_sq_dist + Math.pow(sq_sq_dist, 2)/3);
			}
		}

		return new Matrix(result);
	}

	private double calculateSquareDistance(double[] ds, double[] ds2) {
		double sq_dist = 0;

		for (int i = 0; i < ds.length; i++) {
			double diff = ds[i] - ds2[i];
			sq_dist += diff * diff;
		}

		return sq_dist;
	}

	private double[][] scaleValues(Matrix matrix, double[] loghyper) {
		double[][] array = matrix.getArrayCopy();

		for (int i = 0; i < loghyper.length - 1; i++) {
			//double lengthScale = Math.exp(loghyper[i]);
			double lengthScale = 6.91;

			for (int j = 0; j < array.length; j++) {
				array[j][i] /= lengthScale;
			}
		}

		return array;
	}

	public int getHyperParameterCount(Matrix trainX) {
		return trainX.getColumnDimension() + 1;
	}

}
