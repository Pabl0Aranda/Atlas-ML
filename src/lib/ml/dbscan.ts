import type { Dataset } from '../../types/ml';

export const NOISE = -1;
export const UNVISITED = -2;

export interface DbscanParams {
	/** Neighborhood radius. */
	eps: number;
	/** Neighbors needed, counting the point itself, to be a core point. */
	minPoints: number;
}

export interface DbscanState {
	/** Cluster index per point, or NOISE or UNVISITED. */
	labels: number[];
	core: boolean[];
	visited: boolean[];
	/** Points waiting to be expanded in the current cluster. */
	queue: number[];
	/** The point examined by the latest step. */
	current: number | null;
	clusters: number;
	finished: boolean;
}

export function initDbscan(data: Dataset): DbscanState {
	return {
		labels: data.map(() => UNVISITED),
		core: data.map(() => false),
		visited: data.map(() => false),
		queue: [],
		current: null,
		clusters: 0,
		finished: false,
	};
}

/** Every point within eps of point i, including i itself. */
function neighbors(data: Dataset, i: number, eps: number): number[] {
	const result: number[] = [];
	data.forEach((p, j) => {
		if ((p.x - data[i].x) ** 2 + (p.y - data[i].y) ** 2 <= eps ** 2) result.push(j);
	});
	return result;
}

/** One step examines one point: the next in the cluster, or else the next unvisited. */
export function dbscanStep(state: DbscanState, data: Dataset, { eps, minPoints }: DbscanParams): DbscanState {
	const expanding = state.queue.length > 0;
	const point = expanding ? state.queue[0] : state.visited.indexOf(false);
	if (point < 0) return { ...state, current: null, finished: true };

	const labels = [...state.labels];
	const core = [...state.core];
	const visited = [...state.visited];
	const queue = expanding ? state.queue.slice(1) : [];
	let clusters = state.clusters;

	const around = neighbors(data, point, eps);
	visited[point] = true;
	core[point] = around.length >= minPoints;

	if (core[point]) {
		if (!expanding) labels[point] = clusters++;
		for (const j of around) {
			// Noise can still join as a border point, but it was already examined.
			if (labels[j] === UNVISITED) queue.push(j);
			if (labels[j] === UNVISITED || labels[j] === NOISE) labels[j] = clusters - 1;
		}
	} else if (!expanding) {
		labels[point] = NOISE;
	}

	return { labels, core, visited, queue, current: point, clusters, finished: false };
}

export const clusterCount = (state: DbscanState): number => state.clusters;
