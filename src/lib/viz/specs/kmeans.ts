import type { AlgorithmSpec } from '../../../types/viz';
import { initKMeans, kMeansStep, nearestCentroid, type KMeansState } from '../../ml/kmeans';
import { fixed } from '../scene';

export const kMeans: AlgorithmSpec<KMeansState> = {
	datasets: ['blobs', 'moons', 'circles', 'ellipse'],
	params: [{ key: 'k', label: 'Clusters k', value: 4, min: 2, max: 4, step: 1 }],
	labeled: false,
	speed: 1,
	thumbnailSteps: 12,
	legend: (p) => Array.from({ length: p.k }, (_, i) => `Cluster ${i + 1}`),
	init: (data, p) => initKMeans(data, p.k),
	step: (s, data) => kMeansStep(s, data),
	isDone: (s) => s.converged,
	scene: (s, data) => ({
		region: (x, y) => nearestCentroid(s.centroids, x, y),
		points: data.map((p, i) => ({ x: p.x, y: p.y, cls: s.assignments[i] })),
		markers: s.centroids.map((c, cls) => ({ ...c, cls })),
	}),
	metrics: (s) => [
		{ label: 'Iteration', value: String(s.iteration) },
		{ label: 'Inertia', value: fixed(s.inertia) },
	],
	caption(s) {
		if (s.iteration === 0) return 'k-means++ placed the outlined centroids. Each region is a Voronoi cell.';
		if (s.converged) return `Converged after ${s.iteration} iterations: no point changed cluster.`;
		return 'Each centroid moved to the mean of its points, then every point was reassigned.';
	},
	series: (s) => ({ label: 'Inertia per iteration', values: s.inertiaHistory }),
};
