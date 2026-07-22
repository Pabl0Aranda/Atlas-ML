import type { Dataset } from '../../../types/ml';
import type { AlgorithmSpec } from '../../../types/viz';
import { accuracy } from '../../ml/knn';
import { forestClassify, forestStep, initForest, type ForestState } from '../../ml/randomForest';
import { classPoints, percent } from '../scene';
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
