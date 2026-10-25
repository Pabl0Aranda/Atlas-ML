import type { Dataset, DatasetKind } from '../../types/ml';
import { generateDataset } from './datasets';
import { accuracy, knnClassify } from './knn';
import { trainTestSplit } from './preprocessing';

const SAMPLES = 120;
const TEST_SHARE = 0.25;
/** One random sample gives a noisy curve, so the curve averages many. */
const SEEDS = 20;
/** Rules of thumb, not laws. Each problem needs its own judgment. */
const OVERFIT_GAP = 0.08;
const UNDERFIT_DROP = 0.05;

/** From most flexible to most rigid. A large k averages over almost the whole dataset. */
export const FIT_NEIGHBORS = [1, 3, 5, 9, 15, 25, 41, 61, 75, 85];

export type FitVerdict = 'overfitting' | 'good' | 'underfitting';

export interface FitPoint {
	k: number;
	/** Mean accuracy on the data the model trained on. */
	train: number;
	/** Mean accuracy on data the model never saw. */
	test: number;
}

export function sampleSplit(kind: DatasetKind, seed: number): { train: Dataset; test: Dataset } {
	const data = generateDataset(kind, SAMPLES, seed);
	const isTest = trainTestSplit(data, TEST_SHARE, true, seed);
	return { train: data.filter((_, i) => !isTest[i]), test: data.filter((_, i) => isTest[i]) };
}

/** Training and test accuracy at each level of flexibility, averaged over many samples. */
export function knnFitCurve(kind: DatasetKind): FitPoint[] {
	const samples = Array.from({ length: SEEDS }, (_, i) => sampleSplit(kind, i + 1));
	return FIT_NEIGHBORS.map((k) => {
		const score = (pick: (s: (typeof samples)[number]) => Dataset) =>
			samples.reduce((sum, s) => sum + accuracy(pick(s), (x, y) => knnClassify(s.train, k, x, y)), 0) / SEEDS;
		return { k, train: score((s) => s.train), test: score((s) => s.test) };
	});
}

/** A wide gap means memorizing. A test score far below the best means the model is too simple. */
export function fitVerdict(point: FitPoint, bestTest: number): FitVerdict {
	if (point.train - point.test > OVERFIT_GAP) return 'overfitting';
	return point.test < bestTest - UNDERFIT_DROP ? 'underfitting' : 'good';
}
