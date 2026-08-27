import type { Dataset } from '../../../types/ml';
import type { AlgorithmSpec, Diagram, DiagramNode, SceneLine, ScenePoint } from '../../../types/viz';
import { dbscanStep, initDbscan, NOISE, UNVISITED, type DbscanState } from '../../ml/dbscan';
import { metricBall, PLANE_METRIC_OPTIONS, type PlaneMetric } from '../../ml/distance';
import { hierarchicalStep, initHierarchical, Linkage, type HierarchicalState } from '../../ml/hierarchical';
import { counted, fixed } from '../scene';

/** The plot has four class colors, so smaller clusters only get colors once few remain. */
const COLORED_CLUSTERS = 4;
/** Fifteen nodes is the most a diagram draws with labels. */
const DENDROGRAM_MERGES = 7;

const metricParam = { key: 'metric', label: 'Distance', value: 0, options: PLANE_METRIC_OPTIONS };

const neighborhood = (metric: PlaneMetric, x: number, y: number, radius: number): SceneLine => ({
	tone: 'muted',
	dashed: true,
	path: metricBall(metric, x, y, radius),
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
	if (s.current === null) return 'Every point starts unvisited. Step to examine them one at a time. The dashed outline is the ε neighborhood.';
	if (s.core[s.current]) return `This core point has at least ${minPoints} neighbors, so cluster ${s.labels[s.current] + 1} spreads to all of them.`;
	if (s.labels[s.current] === NOISE) return 'Too few neighbors, and no cluster has reached this point. It is noise for now.';
	return `A border point: it joins cluster ${s.labels[s.current] + 1}, but has too few neighbors to spread it further.`;
}

export const dbscan: AlgorithmSpec<DbscanState> = {
	datasets: ['moons', 'blobs', 'circles'],
	params: [
		{ key: 'eps', label: 'Radius ε', value: 0.15, min: 0.05, max: 0.4, step: 0.01 },
		{ key: 'minPoints', label: 'Min points', value: 4, min: 2, max: 10, step: 1 },
		metricParam,
	],
	labeled: false,
	speed: 1,
	thumbnailSteps: 300,
	legend: () => ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
	init: (data) => initDbscan(data),
	step: (s, data, p) => dbscanStep(s, data, { eps: p.eps, minPoints: p.minPoints, metric: p.metric as PlaneMetric }),
	isDone: (s) => s.finished,
	scene(s, data, p) {
		const current = s.current === null ? null : data[s.current];
		return { points: dbscanPoints(s, data), lines: current ? [neighborhood(p.metric as PlaneMetric, current.x, current.y, p.eps)] : [] };
	},
	metrics: (s, data) => [
		{ label: 'Examined', value: `${s.visited.filter(Boolean).length} of ${data.length}` },
		{ label: 'Clusters', value: String(s.clusters) },
		{ label: 'Core points', value: String(s.core.filter(Boolean).length) },
		{ label: 'Noise', value: String(s.labels.filter((l) => l === NOISE).length) },
	],
	caption: (s, _, p) => dbscanCaption(s, p.minPoints),
};

/** Singletons stay grey until few clusters remain, because early colors would mean nothing. */
function hierarchicalPoints(s: HierarchicalState, data: Dataset): ScenePoint[] {
	const colorAll = s.clusters.length <= COLORED_CLUSTERS;
	const colored = s.clusters.filter((c) => colorAll || c.members.length > 1);
	const points: ScenePoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: 0, muted: true }));
	colored.forEach((cluster, cls) => {
		for (const i of cluster.members) points[i] = { x: data[i].x, y: data[i].y, cls };
	});
	return points;
}

/** The top of the dendrogram. Earlier merges collapse into leaves labeled by their size. */
function dendrogram(s: HierarchicalState): Diagram | undefined {
	const recent = s.merges.slice(-DENDROGRAM_MERGES);
	if (!recent.length) return undefined;
	const top = Math.max(...recent.map((m) => m.height), 1e-9);
	const byId = new Map(recent.map((m) => [m.into, m]));
	const children = new Set(recent.flatMap((m) => [m.a, m.b]));
	const nodes: DiagramNode[] = [];
	const edges: Diagram['edges'] = [];
	let leaves = 0;

	const visit = (id: number, size: number): number => {
		const index = nodes.length;
		const merge = byId.get(id);
		if (!merge) {
			nodes.push({ x: leaves++, y: 1, label: String(size), strength: 0.3 });
			return index;
		}
		nodes.push({ x: 0, y: 1 - merge.height / top, label: merge.height.toFixed(2) });
		const left = visit(merge.a, merge.sizeA);
		const right = visit(merge.b, merge.sizeB);
		edges.push({ from: index, to: left }, { from: index, to: right });
		nodes[index].x = (nodes[left].x + nodes[right].x) / 2;
		return index;
	};
	for (const merge of recent) {
		if (!children.has(merge.into)) visit(merge.into, merge.sizeA + merge.sizeB);
	}
	for (const node of nodes) node.x = leaves > 1 ? node.x / (leaves - 1) : 0.5;
	return { title: 'Latest merges: height is merge distance, leaves show cluster size', nodes, edges };
}

export const hierarchicalClustering: AlgorithmSpec<HierarchicalState> = {
	datasets: ['moons', 'blobs', 'circles'],
	params: [
		{ key: 'linkage', label: 'Linkage', value: Linkage.Single, options: [{ label: 'Single (nearest pair)', value: Linkage.Single }, { label: 'Complete (farthest pair)', value: Linkage.Complete }, { label: 'Average', value: Linkage.Average }] },
		{ key: 'k', label: 'Stop at k clusters', value: 2, min: 1, max: 4, step: 1 },
		metricParam,
	],
	labeled: false,
	speed: 2,
	thumbnailSteps: 118,
	legend: (p) => Array.from({ length: p.k }, (_, i) => `Cluster ${i + 1}`),
	init: (data) => initHierarchical(data),
	step: (s, data, p) => hierarchicalStep(s, data, p.linkage as Linkage, p.metric as PlaneMetric),
	isDone: (s, p) => s.clusters.length <= p.k,
	scene: (s, data) => ({ points: hierarchicalPoints(s, data) }),
	metrics: (s) => [
		{ label: 'Clusters', value: String(s.clusters.length) },
		{ label: 'Merges', value: String(s.merges.length) },
		{ label: 'Latest merge distance', value: s.merges.length ? fixed(s.merges.at(-1)!.height) : '–' },
	],
	caption(s, _, p) {
		const latest = s.merges.at(-1);
		if (!latest) return 'Every point starts as its own cluster. Step to merge the two closest clusters.';
		if (s.clusters.length <= p.k) return `Stopped at ${counted(s.clusters.length, 'cluster')}. A sudden jump in merge distance marks a natural place to cut the tree.`;
		return `Merged clusters of ${latest.sizeA} and ${latest.sizeB} points at distance ${latest.height.toFixed(3)}.`;
	},
	series: (s) => ({ label: 'Merge distance per merge', values: s.merges.map((m) => m.height) }),
	diagram: (s) => dendrogram(s),
};
