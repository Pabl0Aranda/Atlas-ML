import { f1, precision, recall, type Confusion } from './evaluation';

/** Rows are actual classes, and columns are predicted classes. */
export type ConfusionTable = number[][];

export type Normalization = 'none' | 'row' | 'column' | 'all';

export interface ClassScores {
	precision: number;
	recall: number;
	f1: number;
	/** Samples that truly belong to the class. */
	support: number;
}

const ratio = (top: number, bottom: number) => (bottom ? top / bottom : 0);
const total = (m: Confusion) => m.tp + m.fp + m.tn + m.fn;
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);

export const accuracyOf = (m: Confusion): number => ratio(m.tp + m.tn, total(m));

/** Share of real negatives correctly left alone. */
export const specificity = (m: Confusion): number => ratio(m.tn, m.tn + m.fp);

/** Share of negative predictions that were right. */
export const negativePredictiveValue = (m: Confusion): number => ratio(m.tn, m.tn + m.fn);

/** Mean recall of the two classes, so a rare class counts as much as a common one. */
export const balancedAccuracy = (m: Confusion): number => (recall(m) + specificity(m)) / 2;

/** Like F1, but recall counts beta times as much as precision. */
export function fBeta(m: Confusion, beta: number): number {
	if (beta === 1) return f1(m);
	const [p, r] = [precision(m), recall(m)];
	return ratio((1 + beta ** 2) * p * r, beta ** 2 * p + r);
}

/** Correlation between predictions and labels, from -1 to 1. It uses all four cells. */
export function matthews(m: Confusion): number {
	const spread = (m.tp + m.fp) * (m.tp + m.fn) * (m.tn + m.fp) * (m.tn + m.fn);
	return ratio(m.tp * m.tn - m.fp * m.fn, Math.sqrt(spread));
}

/** Agreement with the labels beyond what chance alone would produce. */
export function cohenKappa(m: Confusion): number {
	const n = total(m);
	if (!n) return 0;
	const observed = accuracyOf(m);
	const expected = ((m.tp + m.fp) * (m.tp + m.fn) + (m.tn + m.fn) * (m.tn + m.fp)) / n ** 2;
	return ratio(observed - expected, 1 - expected);
}

/** Total cost of a model's mistakes, when each kind of error has its own price. */
export const expectedCost = (m: Confusion, costs: { fn: number; fp: number }): number => m.fn * costs.fn + m.fp * costs.fp;

export function multiclassMatrix(actual: number[], predicted: number[], classes: number): ConfusionTable {
	const matrix = Array.from({ length: classes }, () => new Array<number>(classes).fill(0));
	actual.forEach((a, i) => matrix[a][predicted[i]]++);
	return matrix;
}

/** Correct predictions sit on the diagonal. */
export const matrixAccuracy = (matrix: ConfusionTable): number => ratio(sum(matrix.map((row, i) => row[i])), sum(matrix.flat()));

/** Recall reads along a row, and precision reads down a column. */
export function perClass(matrix: ConfusionTable): ClassScores[] {
	return matrix.map((row, i) => {
		const support = sum(row);
		const classPrecision = ratio(row[i], sum(matrix.map((r) => r[i])));
		const classRecall = ratio(row[i], support);
		return {
			precision: classPrecision,
			recall: classRecall,
			f1: ratio(2 * classPrecision * classRecall, classPrecision + classRecall),
			support,
		};
	});
}

/** Every class counts equally, so rare classes can pull the score down. */
export function macroF1(matrix: ConfusionTable): number {
	const scores = perClass(matrix);
	return ratio(sum(scores.map((s) => s.f1)), scores.length);
}

/** Each class counts by its size, so a large class can hide a weak small one. */
export function weightedF1(matrix: ConfusionTable): number {
	const scores = perClass(matrix);
	return ratio(sum(scores.map((s) => s.f1 * s.support)), sum(scores.map((s) => s.support)));
}

export function normalizeMatrix(matrix: ConfusionTable, by: Normalization): ConfusionTable {
	if (by === 'none') return matrix;
	const all = sum(matrix.flat());
	return matrix.map((row, i) =>
		row.map((value, j) => {
			if (by === 'all') return ratio(value, all);
			return ratio(value, by === 'row' ? sum(matrix[i]) : sum(matrix.map((r) => r[j])));
		}),
	);
}

/**
 * McNemar's chi-squared statistic, with continuity correction.
 * Takes the samples only the first model got right, and those only the second got right.
 * Values above 3.84 are significant at the 5% level.
 */
export const mcnemar = (firstOnly: number, secondOnly: number): number =>
	ratio(Math.max(0, Math.abs(firstOnly - secondOnly) - 1) ** 2, firstOnly + secondOnly);
