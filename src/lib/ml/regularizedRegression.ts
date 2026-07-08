import type { Dataset } from '../../types/ml';

export interface PenaltyParams {
	/** Overall penalty strength. */
	lambda: number;
	/** Mix of the two penalties: 0 is ridge, 1 is lasso, in between is elastic net. */
	l1Ratio: number;
}

export interface RegularizedState {
	/** Polynomial coefficients. Index 0 is the unpenalized bias. */
	weights: number[];
	pass: number;
	loss: number;
	lossHistory: number[];
	/** Largest coefficient change in the last pass. */
	change: number;
}

const powers = (x: number, degree: number) => Array.from({ length: degree + 1 }, (_, j) => x ** j);

/** (1/2n) * squared error + lambda * (ratio * |w| + (1 - ratio)/2 * w^2). */
export function penalizedLoss(weights: number[], data: Dataset, { lambda, l1Ratio }: PenaltyParams): number {
	const degree = weights.length - 1;
	const error = data.reduce((s, p) => {
		const prediction = powers(p.x, degree).reduce((sum, f, j) => sum + f * weights[j], 0);
		return s + (prediction - p.y) ** 2;
	}, 0);
	const penalized = weights.slice(1);
	const l1 = penalized.reduce((s, w) => s + Math.abs(w), 0);
	const l2 = penalized.reduce((s, w) => s + w * w, 0);
	return error / (2 * Math.max(data.length, 1)) + lambda * (l1Ratio * l1 + ((1 - l1Ratio) / 2) * l2);
}

export function initRegularized(data: Dataset, degree: number, penalty: PenaltyParams): RegularizedState {
	const weights = new Array(degree + 1).fill(0);
	const loss = penalizedLoss(weights, data, penalty);
	return { weights, pass: 0, loss, lossHistory: [loss], change: Infinity };
}

/** Soft thresholding: shrinks toward zero, and returns exactly zero inside the threshold. */
export const softThreshold = (value: number, threshold: number): number =>
	Math.sign(value) * Math.max(Math.abs(value) - threshold, 0);

/** One pass of coordinate descent: each coefficient is solved exactly with the others fixed. */
export function regularizedStep(s: RegularizedState, data: Dataset, penalty: PenaltyParams): RegularizedState {
	const { lambda, l1Ratio } = penalty;
	const n = data.length;
	const degree = s.weights.length - 1;
	const rows = data.map((p) => powers(p.x, degree));
	const weights = [...s.weights];
	const residuals = data.map((p, i) => p.y - rows[i].reduce((sum, f, j) => sum + f * weights[j], 0));

	let change = 0;
	for (let j = 0; j <= degree; j++) {
		let correlation = 0;
		let scale = 0;
		rows.forEach((row, i) => {
			correlation += (row[j] * (residuals[i] + row[j] * weights[j])) / n;
			scale += (row[j] * row[j]) / n;
		});
		const next =
			j === 0
				? correlation / scale
				: softThreshold(correlation, lambda * l1Ratio) / (scale + lambda * (1 - l1Ratio));
		const delta = next - weights[j];
		if (delta !== 0) rows.forEach((row, i) => (residuals[i] -= row[j] * delta));
		weights[j] = next;
		change = Math.max(change, Math.abs(delta));
	}
	const loss = penalizedLoss(weights, data, penalty);
	return { weights, pass: s.pass + 1, loss, lossHistory: [...s.lossHistory, loss], change };
}
