import type { AlgorithmSpec } from '../../../types/viz';
import { PlaneMetric } from '../../ml/distance';
import { initKMeans, kMeansStep, nearestCentroid, type KMeansState } from '../../ml/kmeans';
import { fixed } from '../scene';

export const kMeans: AlgorithmSpec<KMeansState> = {
	datasets: ['blobs', 'moons', 'circles', 'ellipse'],
	params: [
		{ key: 'k', label: 'Clusters k', value: 4, min: 2, max: 4, step: 1 },
		{ key: 'metric', label: 'Distance', value: PlaneMetric.Euclidean, options: [{ label: 'Euclidean (k-means)', value: PlaneMetric.Euclidean }, { label: 'Manhattan (k-medians)', value: PlaneMetric.Manhattan }] },
	],
	labeled: false,
	speed: 1,
	thumbnailSteps: 12,
	legend: (p) => Array.from({ length: p.k }, (_, i) => `Cluster ${i + 1}`),
	init: (data, p) => initKMeans(data, p.k, undefined, p.metric as PlaneMetric),
	step: (s, data, p) => kMeansStep(s, data, p.metric as PlaneMetric),
	isDone: (s) => s.converged,
	scene: (s, data, p) => ({
		region: (x, y) => nearestCentroid(s.centroids, x, y, p.metric as PlaneMetric),
		points: data.map((p, i) => ({ x: p.x, y: p.y, cls: s.assignments[i] })),
		markers: s.centroids.map((c, cls) => ({ ...c, cls })),
	}),
	metrics: (s, _, p) => [
		{ label: 'Iteration', value: String(s.iteration) },
		{ label: p.metric === PlaneMetric.Manhattan ? 'Total distance' : 'Inertia', value: fixed(s.inertia) },
	],
	caption(s, _, p) {
		if (s.iteration === 0) return 'k-means++ placed the outlined centroids. Each region is a Voronoi cell.';
		if (s.converged) return `Converged after ${s.iteration} iterations: no point changed cluster.`;
		return `Each centroid moved to the ${p.metric === PlaneMetric.Manhattan ? 'median' : 'mean'} of its points, then every point was reassigned.`;
	},
	series: (s) => ({ label: 'Cost per iteration', values: s.inertiaHistory }),
};
