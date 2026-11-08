import type { AlgorithmSpec } from '../../../types/viz';
import { PLANE_METRIC_OPTIONS, type PlaneMetric } from '../../ml/distance';
import { knnClassify, knnLeaveOneOut, nearestNeighbors } from '../../ml/knn';
import { classPoints, fixed, percent } from '../scene';

const CLASSES = ['Class A', 'Class B'];
/** Longer neighbor lists are cut here, with the rest counted in the votes. */
const SHOWN = 9;

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
	calculations(s, data, p, probe) {
		const metric = p.metric as PlaneMetric;
		const neighbors = nearestNeighbors(data, s.k, probe.x, probe.y, metric);
		const votes = CLASSES.map((_, c) => neighbors.filter((n) => n.label === c).length);
		const winner = knnClassify(data, s.k, probe.x, probe.y, metric);
		const rows = neighbors.slice(0, SHOWN).map((n, i) => [String(i + 1), `(${n.x.toFixed(2)}, ${n.y.toFixed(2)})`, CLASSES[n.label] ?? String(n.label), fixed(n.distance)]);
		if (neighbors.length > SHOWN) rows.push(['…', `${neighbors.length - SHOWN} more`, '', '']);
		return {
			title: `The ${s.k} nearest ${s.k === 1 ? 'sample' : 'samples'} to the cursor at (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})`,
			columns: ['#', 'Sample', 'Class', 'Distance'],
			rows,
			note: `Votes: ${CLASSES.map((name, c) => `${name} ${votes[c]}`).join(', ')}, so the cursor's location is ${CLASSES[winner]}. Move the cursor over the plot to update the table.`,
		};
	},
};
