import type { Dataset } from '../../types/ml';
import { planeDistance, PlaneMetric } from './distance';

export interface Neighbor {
	x: number;
	y: number;
	label: number;
	distance: number;
}

/** The k samples closest to a location, nearest first. */
export function nearestNeighbors(data: Dataset, k: number, x: number, y: number, metric: PlaneMetric = PlaneMetric.Euclidean): Neighbor[] {
	return data
		.map((p) => ({ x: p.x, y: p.y, label: p.label, distance: planeDistance(metric, p.x, p.y, x, y) }))
		.sort((a, b) => a.distance - b.distance)
		.slice(0, k);
}

/** Majority vote among the k closest points. Ties go to the nearest neighbor's class. */
export function knnClassify(data: Dataset, k: number, x: number, y: number, metric: PlaneMetric = PlaneMetric.Euclidean): number {
	const neighbors = nearestNeighbors(data, k, x, y, metric);
	const votes = new Map<number, number>();
	for (const n of neighbors) votes.set(n.label, (votes.get(n.label) ?? 0) + 1);
	const top = Math.max(...votes.values());
	return neighbors.find((n) => votes.get(n.label) === top)?.label ?? 0;
}

export function accuracy(data: Dataset, classify: (x: number, y: number) => number): number {
	if (!data.length) return 0;
	return data.filter((p) => classify(p.x, p.y) === p.label).length / data.length;
}

/** Leave-one-out accuracy: honest for KNN, where training accuracy at k=1 is always 100%. */
export function knnLeaveOneOut(data: Dataset, k: number, metric: PlaneMetric = PlaneMetric.Euclidean): number {
	if (data.length < 2) return 0;
	const hits = data.filter((p, i) => {
		const rest = data.filter((_, j) => j !== i);
		return knnClassify(rest, k, p.x, p.y, metric) === p.label;
	});
	return hits.length / data.length;
}
