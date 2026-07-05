import type { Dataset } from '../../types/ml';

/** Majority vote among the k closest points. Ties go to the nearest neighbor's class. */
export function knnClassify(data: Dataset, k: number, x: number, y: number): number {
	const neighbors = data
		.map((p) => ({ label: p.label, d: (p.x - x) ** 2 + (p.y - y) ** 2 }))
		.sort((a, b) => a.d - b.d)
		.slice(0, k);
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
export function knnLeaveOneOut(data: Dataset, k: number): number {
	if (data.length < 2) return 0;
	const hits = data.filter((p, i) => {
		const rest = data.filter((_, j) => j !== i);
		return knnClassify(rest, k, p.x, p.y) === p.label;
	});
	return hits.length / data.length;
}
