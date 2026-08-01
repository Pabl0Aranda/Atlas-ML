import type { Dataset } from '../../../types/ml';
import type { AlgorithmSpec, SceneLine, ScenePoint } from '../../../types/viz';
import { dbscanStep, initDbscan, NOISE, UNVISITED, type DbscanState } from '../../ml/dbscan';
import { counted } from '../scene';

const CIRCLE_SEGMENTS = 40;

const ring = (x: number, y: number, radius: number): SceneLine => ({
	tone: 'muted',
	dashed: true,
	path: Array.from({ length: CIRCLE_SEGMENTS + 1 }, (_, i) => {
		const angle = (i / CIRCLE_SEGMENTS) * Math.PI * 2;
		return [x + radius * Math.cos(angle), y + radius * Math.sin(angle)];
	}),
});

function dbscanPoints(s: DbscanState, data: Dataset): ScenePoint[] {
	return data.map((p, i) => {
		const label = s.labels[i];
		const base = { x: p.x, y: p.y, ring: i === s.current };
		if (label === UNVISITED) return { ...base, cls: 0, muted: true, scale: 0.6 };
		if (label === NOISE) return { ...base, cls: 0, muted: true };
		return { ...base, cls: label };
	});
}

function dbscanCaption(s: DbscanState, minPoints: number): string {
	const noise = s.labels.filter((l) => l === NOISE).length;
	if (s.finished) return `Found ${counted(s.clusters, 'cluster')} and ${counted(noise, 'noise point')}, shown in grey. The number of clusters was never an input.`;
	if (s.current === null) return 'Every point starts unvisited. Step to examine them one at a time. The dashed circle is the ε neighborhood.';
	if (s.core[s.current]) return `This core point has at least ${minPoints} neighbors, so cluster ${s.labels[s.current] + 1} spreads to all of them.`;
	if (s.labels[s.current] === NOISE) return 'Too few neighbors, and no cluster has reached this point. It is noise for now.';
	return `A border point: it joins cluster ${s.labels[s.current] + 1}, but has too few neighbors to spread it further.`;
}

export const dbscan: AlgorithmSpec<DbscanState> = {
	datasets: ['moons', 'blobs', 'circles'],
	params: [
		{ key: 'eps', label: 'Radius ε', value: 0.15, min: 0.05, max: 0.4, step: 0.01 },
		{ key: 'minPoints', label: 'Min points', value: 4, min: 2, max: 10, step: 1 },
	],
	labeled: false,
	speed: 1,
	thumbnailSteps: 300,
	legend: () => ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
	init: (data) => initDbscan(data),
	step: (s, data, p) => dbscanStep(s, data, { eps: p.eps, minPoints: p.minPoints }),
	isDone: (s) => s.finished,
	scene(s, data, p) {
		const current = s.current === null ? null : data[s.current];
		return { points: dbscanPoints(s, data), lines: current ? [ring(current.x, current.y, p.eps)] : [] };
	},
	metrics: (s, data) => [
		{ label: 'Examined', value: `${s.visited.filter(Boolean).length} of ${data.length}` },
		{ label: 'Clusters', value: String(s.clusters) },
		{ label: 'Core points', value: String(s.core.filter(Boolean).length) },
		{ label: 'Noise', value: String(s.labels.filter((l) => l === NOISE).length) },
	],
	caption: (s, _, p) => dbscanCaption(s, p.minPoints),
};
