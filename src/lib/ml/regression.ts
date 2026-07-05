import type { Dataset } from '../../types/ml';

/** Model state for polynomial regression. Degree 1 is linear regression. */
export interface RegressionState {
	/** Coefficients w0..wd of w0 + w1 x + ... + wd x^d. */
	weights: number[];
	epoch: number;
	loss: number;
	lossHistory: number[];
}

export function predict(weights: number[], x: number): number {
	// Horner's method evaluates the polynomial in d multiplications.
	return weights.reduceRight((acc, w) => acc * x + w, 0);
}

/** Mean squared error plus an L2 penalty on every weight except the bias. */
export function regressionLoss(weights: number[], data: Dataset, l2 = 0): number {
	const mse = data.reduce((s, p) => s + (predict(weights, p.x) - p.y) ** 2, 0) / Math.max(data.length, 1);
	return mse + l2 * weights.slice(1).reduce((s, w) => s + w * w, 0);
}

export function initRegression(data: Dataset, degree: number, l2 = 0): RegressionState {
	const weights = new Array(degree + 1).fill(0);
	const loss = regressionLoss(weights, data, l2);
	return { weights, epoch: 0, loss, lossHistory: [loss] };
}

/** One full-batch gradient descent update on the regularized MSE. */
export function regressionStep(
	state: RegressionState,
	data: Dataset,
	learningRate: number,
	l2 = 0,
): RegressionState {
	const gradient = state.weights.map(() => 0);
	for (const p of data) {
		const error = predict(state.weights, p.x) - p.y;
		let power = 1;
		for (let j = 0; j < gradient.length; j++) {
			gradient[j] += (2 * error * power) / data.length;
			power *= p.x;
		}
	}
	const weights = state.weights.map((w, j) => w - learningRate * (gradient[j] + (j > 0 ? 2 * l2 * w : 0)));
	const loss = regressionLoss(weights, data, l2);
	return { weights, epoch: state.epoch + 1, loss, lossHistory: [...state.lossHistory, loss] };
}
