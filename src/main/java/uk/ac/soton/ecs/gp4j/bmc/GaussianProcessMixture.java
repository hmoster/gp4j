package uk.ac.soton.ecs.gp4j.bmc;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.Validate;

import uk.ac.soton.ecs.gp4j.gp.GaussianPredictor;
import uk.ac.soton.ecs.gp4j.gp.GaussianProcess;
import uk.ac.soton.ecs.gp4j.gp.GaussianProcessPrediction;
import Jama.Matrix;

public class GaussianProcessMixture implements
		GaussianPredictor<GaussianProcessMixturePrediction> {

	private final List<GaussianProcess> gaussianProcesses;

	private final List<Double> weights;

	protected GaussianProcessMixture(List<GaussianProcess> gaussianProcesses,
			List<Double> weights) {
		//Validate.isTrue(gaussianProcesses.size() == weights.size());

		this.gaussianProcesses = gaussianProcesses;
		this.weights = weights;
	}

	protected GaussianProcessMixture() {
		this.gaussianProcesses = new ArrayList<GaussianProcess>();
		this.weights = new ArrayList<Double>();
	}

	protected void addGaussianProcess(GaussianProcess process, double weight) {
		gaussianProcesses.add(process);
		weights.add(weight);
	}

	public double getWeight(GaussianProcess process) {
		return weights.get(gaussianProcesses.indexOf(process));
	}

	public List<GaussianProcess> getGaussianProcesses() {
		return gaussianProcesses;
	}

	public GaussianProcessMixturePrediction calculatePrediction(Matrix testX) {
		Matrix resultMean = new Matrix(testX.getRowDimension(), 1);
		Matrix resultVariance = new Matrix(testX.getRowDimension(), 1);
        double yTrainStd = 0.0;

		// marginalize the gaussianprocess mixture using equations
		// 3.8.18 and 3.8.19
        if (gaussianProcesses.size() > 1) {
			for (int i = 0; i < gaussianProcesses.size(); i++) {
				GaussianProcess process = gaussianProcesses.get(i);

				GaussianProcessPrediction prediction = process
						.calculatePrediction(testX);
				double weight = weights.get(i);
				Matrix mean = prediction.getMean();
				Matrix variance = prediction.getVariance();

				resultMean.plusEquals(mean.times(weight));

				Matrix meanSq = mean.arrayTimes(mean);

				resultVariance.plusEquals(variance.plus(meanSq).times(weight));
			}

			resultVariance.minusEquals(resultMean.arrayTimes(resultMean));
		} else if(gaussianProcesses.size() == 1) {
			GaussianProcess process = gaussianProcesses.get(0);

			GaussianProcessPrediction prediction = process
					.calculatePrediction(testX);
			return new GaussianProcessMixturePrediction(testX, prediction.getMean(),
					prediction.getVariance(), process.getStd());
		}

		return new GaussianProcessMixturePrediction(testX, resultMean,
				resultVariance, 1.0);
	}

	public Matrix getTrainX() {
		return getGaussianProcesses().get(0).getTrainX();
	}

	public Matrix getTrainY() {
		return getGaussianProcesses().get(0).getTrainY();
	}
}
