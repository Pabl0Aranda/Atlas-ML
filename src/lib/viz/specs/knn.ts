import type { AlgorithmSpec } from '../../../types/viz';
import { PLANE_METRIC_OPTIONS, type PlaneMetric } from '../../ml/distance';
import { knnClassify, knnLeaveOneOut } from '../../ml/knn';
import { classPoints, percent } from '../scene';

const MAX_K = 41;

interface State {
	k: number;
}

export const knn: AlgorithmSpec<State> = {
	datasets: ['moons', 'circles', 'xor', 'outliers', 'linear'],
	params: [
		{ key: 'k', label: 'Neighbors k', value: 1, min: 1, max: MAX_K, step: 2 },
		{ key: 'metric', label: 'Distance', value: 0, live: true, options: PLANE_METRIC_OPTIONS },
	],
	labeled: true,
	holdout: true,
	speed: 1,
	thumbnailSteps: 2,
	legend: () => ['Class A', 'Class B'],
	init: (_, p) => ({ k: p.k }),
	step: (s) => ({ k: s.k + 2 }),
	isDone: (s) => s.k >= MAX_K,
	scene: (s, data, p) => ({ region: (x, y) => knnClassify(data, s.k, x, y, p.metric as PlaneMetric), points: classPoints(data) }),
	metrics: (s, data, p) => [
		{ label: 'k', value: String(s.k) },
		{ label: 'Leave-one-out accuracy', value: percent(knnLeaveOneOut(data, s.k, p.metric as PlaneMetric)) },
	],
	caption: (s) =>
		s.k === 1
			? 'With k = 1 the boundary is the Voronoi diagram of the samples. Step to raise k and smooth it.'
			: `Each location takes the majority vote of its ${s.k} nearest samples.`,
};
