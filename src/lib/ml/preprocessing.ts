import type { Dataset, DataPoint } from '../../types/ml';
import { mean } from './linalg';
import { createRng } from './random';

/* Cleaning */

/** Linear-interpolated quantile, with q between 0 and 1. */
export function quantile(values: number[], q: number): number {
	const sorted = [...values].sort((a, b) => a - b);
	const position = (sorted.length - 1) * q;
	const low = Math.floor(position);
	const high = Math.ceil(position);
	return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

/** Tukey's fences: values outside these bounds count as outliers. */
export function iqrBounds(values: number[], k = 1.5): [number, number] {
	const q1 = quantile(values, 0.25);
	const q3 = quantile(values, 0.75);
	return [q1 - k * (q3 - q1), q3 + k * (q3 - q1)];
}

/** Flags each point that falls outside the fences on either axis. */
export function flagOutliers(data: Dataset, k = 1.5): boolean[] {
	const [xLow, xHigh] = iqrBounds(data.map((p) => p.x), k);
	const [yLow, yHigh] = iqrBounds(data.map((p) => p.y), k);
	return data.map((p) => p.x < xLow || p.x > xHigh || p.y < yLow || p.y > yHigh);
}

/** Fills missing entries with the median of the observed values. */
export function imputeMedian(values: (number | null)[]): number[] {
	const observed = values.filter((v): v is number => v !== null);
	const fill = quantile(observed, 0.5);
	return values.map((v) => v ?? fill);
}

export function dropDuplicates(data: Dataset): Dataset {
	const seen = new Set<string>();
	return data.filter((p) => {
		const key = `${p.x}|${p.y}|${p.label}`;
		if (seen.has(key)) return false;
		seen.add(key);
		return true;
	});
}

/* Processing */

export type Scaler = (value: number) => number;

/** Learns min and max from the data it is given, and maps them to 0 and 1. */
export function fitMinMax(values: number[]): Scaler {
	const min = Math.min(...values);
	const range = Math.max(...values) - min || 1;
	return (value) => (value - min) / range;
}

/** Learns mean and standard deviation, and maps values to z-scores. */
export function fitStandard(values: number[]): Scaler {
	const center = mean(values);
	const deviation = standardDeviation(values) || 1;
	return (value) => (value - center) / deviation;
}

export function standardDeviation(values: number[]): number {
	const center = mean(values);
	return Math.sqrt(mean(values.map((v) => (v - center) ** 2)));
}

/** Turns a categorical column into one binary column per category. */
export function oneHot(values: string[]): { categories: string[]; rows: number[][] } {
	const categories = [...new Set(values)].sort();
	return { categories, rows: values.map((v) => categories.map((c) => (c === v ? 1 : 0))) };
}

/* Separating */

function shuffle<T>(items: T[], rng: () => number): T[] {
	const result = [...items];
	for (let i = result.length - 1; i > 0; i--) {
		const j = Math.floor(rng() * (i + 1));
		[result[i], result[j]] = [result[j], result[i]];
	}
	return result;
}

/**
 * Returns a flag per sample that is true for the test set.
 * Stratifying splits each class separately, which preserves class proportions.
 */
export function trainTestSplit(data: DataPoint[], testRatio: number, stratified: boolean, seed = 21): boolean[] {
	const rng = createRng(seed);
	const groups = new Map<number, number[]>();
	data.forEach((p, i) => {
		const key = stratified ? p.label : 0;
		groups.set(key, [...(groups.get(key) ?? []), i]);
	});
	const isTest = data.map(() => false);
	for (const indices of groups.values()) {
		const count = Math.round(indices.length * testRatio);
		for (const i of shuffle(indices, rng).slice(0, count)) isTest[i] = true;
	}
	return isTest;
}

/** Assigns every sample to one of k folds of near-equal size. */
export function kFold(n: number, k: number, seed = 21): number[][] {
	const folds: number[][] = Array.from({ length: k }, () => []);
	shuffle(Array.from({ length: n }, (_, i) => i), createRng(seed)).forEach((index, i) => folds[i % k].push(index));
	return folds;
}
