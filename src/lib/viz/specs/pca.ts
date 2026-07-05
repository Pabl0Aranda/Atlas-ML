import type { AlgorithmSpec, SceneLine } from '../../../types/viz';
import { eigenvalues2d, initPca, pcaStep, project, type PcaState } from '../../ml/pca';
import { percent } from '../scene';

const MAX_ITERATIONS = 60;

function throughCenter(s: PcaState, [u, v]: [number, number], dashed: boolean): SceneLine {
	const reach = 3;
	return {
		tone: 'ink',
		dashed,
		path: [
			[s.center.x - u * reach, s.center.y - v * reach],
			[s.center.x + u * reach, s.center.y + v * reach],
		],
	};
}

const best = (s: PcaState) => {
	const [l1, l2] = eigenvalues2d(s.covariance);
	return l1 / Math.max(l1 + l2, 1e-12);
};

export const pca: AlgorithmSpec<PcaState> = {
	datasets: ['ellipse', 'trend', 'blobs', 'circles'],
	params: [
		{ key: 'angle', label: 'Starting angle (rad)', value: 2.4, min: 0, max: 3.1, step: 0.1 },
	],
	labeled: false,
	speed: 1,
	thumbnailSteps: 30,
	legend: () => ['Samples'],
	init: (data, p) => initPca(data, p.angle),
	step: (s) => pcaStep(s),
	isDone: (s) => s.iteration >= MAX_ITERATIONS || best(s) - s.explained < 1e-6,
	scene: (s, data) => ({
		points: data.map((p) => ({ x: p.x, y: p.y, cls: 0 })),
		lines: [
			...data.map((p): SceneLine => {
				const q = project(s, p.x, p.y);
				return { tone: 'muted', path: [[p.x, p.y], [q.x, q.y]] };
			}),
			throughCenter(s, s.axis, false),
			throughCenter(s, [-s.axis[1], s.axis[0]], true),
		],
	}),
	metrics: (s) => [
		{ label: 'Iteration', value: String(s.iteration) },
		{ label: 'Variance on axis', value: percent(s.explained) },
		{ label: 'Best possible', value: percent(best(s)) },
	],
	caption: (s) =>
		s.iteration === 0
			? 'A random axis through the mean. Thin lines project each sample onto it.'
			: `Power iteration rotated the axis: it now captures ${percent(s.explained)} of the variance.`,
	series: (s) => ({ label: 'Explained variance', values: s.explainedHistory }),
};
