import type { Dataset } from '../../types/ml';
import { accuracy, knnClassify } from './knn';
import { mean } from './linalg';
import { kFold, quantile } from './preprocessing';
import { createRng } from './random';

/* Validation */

/** Trains on all folds but one, scores on the held-out fold. */
export function foldScore(data: Dataset, folds: number[][], fold: number, neighbors: number): number {
	const held = new Set(folds[fold]);
	const train = data.filter((_, i) => !held.has(i));
	return accuracy(data.filter((_, i) => held.has(i)), (x, y) => knnClassify(train, neighbors, x, y));
}

/** Mean k-fold accuracy of a KNN classifier. */
export function crossValidate(data: Dataset, k: number, neighbors: number, seed = 21): number {
	const folds = kFold(data.length, k, seed);
	return mean(folds.map((_, fold) => foldScore(data, folds, fold, neighbors)));
}

/* Testing: classification */

export interface Confusion {
	tp: number;
	fp: number;
	tn: number;
	fn: number;
}

export function confusionMatrix(actual: number[], predicted: number[]): Confusion {
	const m = { tp: 0, fp: 0, tn: 0, fn: 0 };
	actual.forEach((a, i) => {
		if (predicted[i] === 1) a === 1 ? m.tp++ : m.fp++;
		else a === 1 ? m.fn++ : m.tn++;
	});
	return m;
}

const ratio = (top: number, bottom: number) => (bottom ? top / bottom : 0);

export const precision = (m: Confusion): number => ratio(m.tp, m.tp + m.fp);
export const recall = (m: Confusion): number => ratio(m.tp, m.tp + m.fn);
export const f1 = (m: Confusion): number => ratio(2 * precision(m) * recall(m), precision(m) + recall(m));

/** ROC AUC: the chance a random positive scores above a random negative. */
export function rocAuc(actual: number[], scores: number[]): number {
	const positives = scores.filter((_, i) => actual[i] === 1);
	const negatives = scores.filter((_, i) => actual[i] !== 1);
	let wins = 0;
	for (const p of positives) for (const n of negatives) wins += p > n ? 1 : p === n ? 0.5 : 0;
	return ratio(wins, positives.length * negatives.length);
}

/* Testing: regression */

export const mae = (actual: number[], predicted: number[]): number =>
	mean(actual.map((a, i) => Math.abs(a - predicted[i])));

export const rmse = (actual: number[], predicted: number[]): number =>
	Math.sqrt(mean(actual.map((a, i) => (a - predicted[i]) ** 2)));

/** Share of variance explained. Zero matches always predicting the mean. */
export function r2(actual: number[], predicted: number[]): number {
	const center = mean(actual);
	const residual = actual.reduce((s, a, i) => s + (a - predicted[i]) ** 2, 0);
	const total = actual.reduce((s, a) => s + (a - center) ** 2, 0);
	return total ? 1 - residual / total : 0;
}

/* Testing: uncertainty */

/** Percentile bootstrap interval for the mean of per-sample scores. */
export function bootstrapInterval(values: number[], rounds = 1000, level = 0.95, seed = 31): [number, number] {
	const rng = createRng(seed);
	const means = Array.from({ length: rounds }, () =>
		mean(values.map(() => values[Math.floor(rng() * values.length)])),
	);
	return [quantile(means, (1 - level) / 2), quantile(means, (1 + level) / 2)];
}
