import type { Dataset } from '../../types/ml';
import { dot } from './linalg';

export interface LogisticState {
	weights: number[];
	epoch: number;
	loss: number;
	lossHistory: number[];
}

/** Degree 1 gives a straight boundary. Degree 2 adds squares and the interaction term. */
export function logisticFeatures(x: number, y: number, degree: number): number[] {
	return degree === 1 ? [1, x, y] : [1, x, y, x * x, y * y, x * y];
}

export const logisticProbability = (weights: number[], x: number, y: number, degree: number): number =>
	1 / (1 + Math.exp(-dot(weights, logisticFeatures(x, y, degree))));

/** Mean negative log-likelihood plus an L2 penalty on everything except the bias. */
export function logisticLoss(weights: number[], data: Dataset, degree: number, l2 = 0): number {
	const eps = 1e-12;
	const nll = data.reduce((s, p) => {
		const q = logisticProbability(weights, p.x, p.y, degree);
		return s - Math.log((p.label ? q : 1 - q) + eps);
	}, 0);
	return nll / Math.max(data.length, 1) + (l2 / 2) * weights.slice(1).reduce((s, w) => s + w * w, 0);
}

export function initLogistic(data: Dataset, degree: number, l2 = 0): LogisticState {
	const weights = logisticFeatures(0, 0, degree).map(() => 0);
	const loss = logisticLoss(weights, data, degree, l2);
	return { weights, epoch: 0, loss, lossHistory: [loss] };
}

/** One full-batch gradient step. The gradient per sample is (prediction - label) * features. */
export function logisticStep(s: LogisticState, data: Dataset, degree: number, rate: number, l2 = 0): LogisticState {
	const gradient = s.weights.map(() => 0);
	for (const p of data) {
		const error = logisticProbability(s.weights, p.x, p.y, degree) - p.label;
		logisticFeatures(p.x, p.y, degree).forEach((f, j) => (gradient[j] += (error * f) / data.length));
	}
	const weights = s.weights.map((w, j) => w - rate * (gradient[j] + (j > 0 ? l2 * w : 0)));
	const loss = logisticLoss(weights, data, degree, l2);
	return { weights, epoch: s.epoch + 1, loss, lossHistory: [...s.lossHistory, loss] };
}
