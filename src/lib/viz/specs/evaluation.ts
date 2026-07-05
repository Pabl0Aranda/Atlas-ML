import type { Dataset } from '../../../types/ml';
import type { AlgorithmSpec } from '../../../types/viz';
import { confusionMatrix, crossValidate, f1, foldScore, precision, recall, rocAuc } from '../../ml/evaluation';
import { accuracy, knnClassify } from '../../ml/knn';
import { mean } from '../../ml/linalg';
import { kFold, standardDeviation } from '../../ml/preprocessing';
import { classPoints, percent } from '../scene';

interface CvState {
	folds: number[][];
	current: number;
	scores: number[];
}

export const crossValidation: AlgorithmSpec<CvState> = {
	datasets: ['outliers', 'moons', 'circles', 'xor'],
	params: [
		{ key: 'folds', label: 'Folds', value: 5, min: 2, max: 10, step: 1 },
		{ key: 'neighbors', label: 'KNN neighbors', value: 5, min: 1, max: 31, step: 2 },
	],
	labeled: true,
	speed: 1,
	thumbnailSteps: 1,
	legend: () => ['Class A', 'Class B'],
	init: (data, p) => ({ folds: kFold(data.length, p.folds), current: -1, scores: [] }),
	step(s, data, p) {
		const current = s.current + 1;
		return { ...s, current, scores: [...s.scores, foldScore(data, s.folds, current, p.neighbors)] };
	},
	isDone: (s) => s.current >= s.folds.length - 1,
	scene(s, data, p) {
		if (s.current < 0) return { points: classPoints(data) };
		const held = new Set(s.folds[s.current]);
		const train = data.filter((_, i) => !held.has(i));
		return {
			region: (x, y) => knnClassify(train, p.neighbors, x, y),
			points: data.map((d, i) => ({ x: d.x, y: d.y, cls: d.label, ring: held.has(i) })),
		};
	},
	metrics: (s) => [
		{ label: 'Fold', value: `${s.current + 1} / ${s.folds.length}` },
		{ label: 'This fold', value: s.scores.length ? percent(s.scores.at(-1)!) : '–' },
		{ label: 'Mean accuracy', value: s.scores.length ? percent(mean(s.scores)) : '–' },
		{ label: 'Std across folds', value: s.scores.length > 1 ? percent(standardDeviation(s.scores)) : '–' },
	],
	caption(s) {
		if (s.current < 0) return 'Step to train on all folds but one, and score on the one held out.';
		return 'Ringed points are this round\'s validation fold. The shaded boundary was trained without them.';
	},
	series: (s) => ({ label: 'Accuracy per fold', values: s.scores }),
};

const MAX_NEIGHBORS = 41;

interface SearchState {
	k: number;
	tried: { k: number; score: number }[];
}

const bestTrial = (s: SearchState) => s.tried.reduce((a, b) => (b.score > a.score ? b : a));

export const hyperparameterSearch: AlgorithmSpec<SearchState> = {
	datasets: ['outliers', 'moons', 'xor', 'circles'],
	params: [{ key: 'folds', label: 'Folds', value: 5, min: 2, max: 10, step: 1 }],
	labeled: true,
	speed: 1,
	thumbnailSteps: 3,
	legend: () => ['Class A', 'Class B'],
	init: (data, p) => ({ k: 1, tried: [{ k: 1, score: crossValidate(data, p.folds, 1) }] }),
	step(s, data, p) {
		const k = s.k + 2;
		return { k, tried: [...s.tried, { k, score: crossValidate(data, p.folds, k) }] };
	},
	isDone: (s) => s.k >= MAX_NEIGHBORS,
	scene: (s, data) => ({ region: (x, y) => knnClassify(data, s.k, x, y), points: classPoints(data) }),
	metrics: (s, data) => [
		{ label: 'Trying k', value: String(s.k) },
		{ label: 'Training accuracy', value: percent(accuracy(data, (x, y) => knnClassify(data, s.k, x, y))) },
		{ label: 'Validation accuracy', value: percent(s.tried.at(-1)!.score) },
		{ label: 'Best k so far', value: `${bestTrial(s).k} (${percent(bestTrial(s).score)})` },
	],
	caption: (s) =>
		s.k === 1
			? 'At k = 1 training accuracy is perfect, yet validation accuracy is what counts. Step through the grid.'
			: `Grid search over k. Each value is scored by cross-validation, never by the test set.`,
	series: (s) => ({ label: 'Validation accuracy per candidate', values: s.tried.map((t) => t.score) }),
};

const SCORE_SHARPNESS = 3;
const THRESHOLD_STEP = 0.05;

const probability = (x: number, y: number) => 1 / (1 + Math.exp(-SCORE_SHARPNESS * (x + y)));

interface ThresholdState {
	threshold: number;
	f1History: number[];
}

function evaluate(data: Dataset, threshold: number) {
	const actual = data.map((d) => d.label);
	const scores = data.map((d) => probability(d.x, d.y));
	const matrix = confusionMatrix(actual, scores.map((s) => (s >= threshold ? 1 : 0)));
	return { matrix, auc: rocAuc(actual, scores) };
}

export const thresholdMetrics: AlgorithmSpec<ThresholdState> = {
	datasets: ['outliers', 'imbalanced', 'linear'],
	params: [{ key: 'threshold', label: 'Starting threshold', value: 0.1, min: 0.05, max: 0.9, step: 0.05 }],
	labeled: true,
	speed: 1,
	thumbnailSteps: 8,
	legend: () => ['Negative (A)', 'Positive (B)'],
	init: (data, p) => ({ threshold: p.threshold, f1History: [f1(evaluate(data, p.threshold).matrix)] }),
	step(s, data) {
		const threshold = Math.round((s.threshold + THRESHOLD_STEP) * 100) / 100;
		return { threshold, f1History: [...s.f1History, f1(evaluate(data, threshold).matrix)] };
	},
	isDone: (s) => s.threshold >= 0.95,
	scene: (s, data) => ({
		region: (x, y) => (probability(x, y) >= s.threshold ? 1 : 0),
		points: classPoints(data),
	}),
	metrics(s, data) {
		const { matrix: m, auc } = evaluate(data, s.threshold);
		return [
			{ label: 'Threshold', value: s.threshold.toFixed(2) },
			{ label: 'TP / FP', value: `${m.tp} / ${m.fp}` },
			{ label: 'FN / TN', value: `${m.fn} / ${m.tn}` },
			{ label: 'Accuracy', value: percent((m.tp + m.tn) / Math.max(data.length, 1)) },
			{ label: 'Precision', value: percent(precision(m)) },
			{ label: 'Recall', value: percent(recall(m)) },
			{ label: 'F1', value: percent(f1(m)) },
			{ label: 'ROC AUC', value: auc.toFixed(3) },
		];
	},
	caption: (s) =>
		`Samples in the shaded B region are predicted positive at threshold ${s.threshold.toFixed(2)}. Raising it trades recall for precision.`,
	series: (s) => ({ label: 'F1 per threshold tried', values: s.f1History }),
};
