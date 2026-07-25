import type { Dataset } from '../../../types/ml';
import type { AlgorithmSpec } from '../../../types/viz';
import { boostingClassify, boostingProbability, boostingStep, initBoosting, type BoostingState } from '../../ml/gradientBoosting';
import { accuracy } from '../../ml/knn';
import { forestClassify, forestStep, initForest, type ForestState } from '../../ml/randomForest';
import { classPoints, fixed, percent } from '../scene';
import { treeDiagram } from './decisionTree';

const LEGEND = ['Class A', 'Class B'];
const DATASETS: AlgorithmSpec['datasets'] = ['moons', 'circles', 'xor', 'outliers', 'linear'];

const forestAccuracy = (s: ForestState, data: Dataset) => accuracy(data, (x, y) => forestClassify(s, x, y));

export const randomForest: AlgorithmSpec<ForestState> = {
	datasets: DATASETS,
	params: [
		{ key: 'trees', label: 'Trees', value: 25, min: 1, max: 60, step: 1 },
		{ key: 'maxDepth', label: 'Max depth', value: 5, min: 1, max: 8, step: 1 },
	],
	labeled: true,
	speed: 1,
	thumbnailSteps: 15,
	legend: () => LEGEND,
	init: (data) => initForest(data),
	step: (s, data, p) => forestStep(s, data, p.maxDepth),
	isDone: (s, p) => s.trees.length >= p.trees,
	scene: (s, data) => ({ region: (x, y) => forestClassify(s, x, y), points: classPoints(data) }),
	metrics: (s, data) => [
		{ label: 'Trees', value: String(s.trees.length) },
		{ label: 'Training accuracy', value: s.trees.length ? percent(forestAccuracy(s, data)) : '–' },
		{ label: 'Out-of-bag accuracy', value: s.trees.length ? percent(s.oobAccuracy) : '–' },
	],
	caption(s, data) {
		if (!s.trees.length) return 'No trees yet. Step to grow one on a bootstrap sample of the data.';
		if (s.trees.length === 1) return 'One tree on one bootstrap sample gives a jagged, overconfident boundary.';
		return `${s.trees.length} trees vote. Training accuracy is ${percent(forestAccuracy(s, data))}, but out-of-bag accuracy of ${percent(s.oobAccuracy)} is the honest score.`;
	},
	series: (s) => ({ label: 'Out-of-bag accuracy per tree added', values: s.oobHistory }),
	diagram(s) {
		const latest = s.trees.at(-1);
		return latest && { ...treeDiagram(latest), title: `Tree ${s.trees.length}, the latest to join the vote` };
	},
};

const boostingAccuracy = (s: BoostingState, data: Dataset) => accuracy(data, (x, y) => boostingClassify(s, x, y));

export const gradientBoosting: AlgorithmSpec<BoostingState> = {
	datasets: DATASETS,
	params: [
		{ key: 'rounds', label: 'Trees', value: 60, min: 1, max: 150, step: 1 },
		{ key: 'learningRate', label: 'Learning rate α', value: 0.3, min: 0.01, max: 1, step: 0.01 },
		{ key: 'maxDepth', label: 'Tree depth', value: 2, min: 1, max: 4, step: 1 },
	],
	labeled: true,
	speed: 1,
	thumbnailSteps: 30,
	legend: () => LEGEND,
	init: (data) => initBoosting(data),
	step: (s, data, p) => boostingStep(s, data, { learningRate: p.learningRate, maxDepth: p.maxDepth }),
	isDone: (s, p) => s.trees.length >= p.rounds,
	scene: (s, data) => ({ region: (x, y) => boostingClassify(s, x, y), points: classPoints(data) }),
	metrics: (s, data) => [
		{ label: 'Trees', value: String(s.trees.length) },
		{ label: 'Log loss', value: fixed(s.loss) },
		{ label: 'Training accuracy', value: percent(boostingAccuracy(s, data)) },
	],
	caption(s) {
		if (!s.trees.length) return 'The model starts from the overall class balance. Step to add a tree that fits the remaining errors.';
		return `Tree ${s.trees.length} fits the gap between each label and the current probability, then adds a small correction.`;
	},
	series: (s) => ({ label: 'Log loss per tree added', values: s.lossHistory }),
	describe: (s, probe) => `(${probe.x.toFixed(2)}, ${probe.y.toFixed(2)}) → P(B) = ${boostingProbability(s, probe.x, probe.y).toFixed(2)}`,
};
