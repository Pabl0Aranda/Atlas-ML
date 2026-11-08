import type { Dataset, DataPoint } from '../../types/ml';
import { logisticFeatures, logisticProbability } from './logisticRegression';
import { createRng } from './random';

/** The class with fewer samples, which is usually the one that matters, such as fraud. */
export const MINORITY = 1;

const byClass = (data: Dataset) => ({
	minority: data.filter((p) => p.label === MINORITY),
	majority: data.filter((p) => p.label !== MINORITY),
});

/** Shuffles a copy with Fisher-Yates, so every run with the same seed is identical. */
function shuffled<T>(items: T[], rng: () => number): T[] {
	const copy = [...items];
	for (let i = copy.length - 1; i > 0; i--) {
		const j = Math.floor(rng() * (i + 1));
		[copy[i], copy[j]] = [copy[j], copy[i]];
	}
	return copy;
}

/** Random undersampling: keeps every minority sample and as many majority samples, chosen at random. */
export function undersample(data: Dataset, seed = 5): { kept: Dataset; removed: Dataset } {
	const { minority, majority } = byClass(data);
	const order = shuffled(majority, createRng(seed));
	return { kept: [...order.slice(0, minority.length), ...minority], removed: order.slice(minority.length) };
}

/** Random oversampling: copies minority samples, chosen at random, until both classes are the same size. */
export function oversample(data: Dataset, seed = 5): Dataset {
	const { minority, majority } = byClass(data);
	if (!minority.length) return [];
	const rng = createRng(seed);
	return Array.from({ length: Math.max(0, majority.length - minority.length) }, () => ({ ...minority[Math.floor(rng() * minority.length)] }));
}

export interface SyntheticSample {
	point: DataPoint;
	/** The two real minority samples it was placed between. */
	from: DataPoint;
	to: DataPoint;
}

/** The k nearest other members of a group, by Euclidean distance. */
export function nearest(group: Dataset, of: DataPoint, k: number): Dataset {
	return group
		.filter((p) => p !== of)
		.map((p) => ({ p, d: (p.x - of.x) ** 2 + (p.y - of.y) ** 2 }))
		.sort((a, b) => a.d - b.d)
		.slice(0, k)
		.map((e) => e.p);
}

/**
 * SMOTE: each new sample lies at a random point on the segment between a random minority
 * sample and one of its k nearest minority neighbors. Unlike copies, the new samples fill
 * in the space between real ones. Adds samples until both classes are the same size.
 */
export function smote(data: Dataset, k: number, seed = 5): SyntheticSample[] {
	const { minority, majority } = byClass(data);
	if (minority.length < 2) return [];
	const rng = createRng(seed);
	return Array.from({ length: Math.max(0, majority.length - minority.length) }, () => {
		const from = minority[Math.floor(rng() * minority.length)];
		const neighbors = nearest(minority, from, k);
		const to = neighbors[Math.floor(rng() * neighbors.length)];
		const t = rng();
		return { point: { x: from.x + t * (to.x - from.x), y: from.y + t * (to.y - from.y), label: MINORITY }, from, to };
	});
}

/** Class weights that make each class count equally in the loss: n / (2 · samples in the class). */
export function classWeights(data: Dataset): [number, number] {
	const { minority, majority } = byClass(data);
	const weight = (count: number) => (count ? data.length / (2 * count) : 0);
	return [weight(majority.length), weight(minority.length)];
}

/** One full-batch gradient step of logistic regression, with each sample's error scaled by its class weight. */
export function weightedLogisticStep(weights: number[], data: Dataset, classWeight: [number, number], rate: number): number[] {
	const gradient = weights.map(() => 0);
	const total = data.reduce((s, p) => s + classWeight[p.label], 0) || 1;
	for (const p of data) {
		const error = (logisticProbability(weights, p.x, p.y, 1) - p.label) * classWeight[p.label];
		logisticFeatures(p.x, p.y, 1).forEach((f, j) => (gradient[j] += (error * f) / total));
	}
	return weights.map((w, j) => w - rate * gradient[j]);
}

export interface MinorityScores {
	accuracy: number;
	/** The share of real minority samples the model catches. */
	recall: number;
	/** The share of minority predictions that are right. */
	precision: number;
}

export function minorityScores(predict: (x: number, y: number) => number, test: Dataset): MinorityScores {
	let tp = 0;
	let fp = 0;
	let fn = 0;
	let correct = 0;
	for (const p of test) {
		const guess = predict(p.x, p.y);
		if (guess === p.label) correct++;
		if (guess === MINORITY && p.label === MINORITY) tp++;
		if (guess === MINORITY && p.label !== MINORITY) fp++;
		if (guess !== MINORITY && p.label === MINORITY) fn++;
	}
	return { accuracy: test.length ? correct / test.length : 0, recall: tp + fn ? tp / (tp + fn) : 0, precision: tp + fp ? tp / (tp + fp) : 0 };
}
