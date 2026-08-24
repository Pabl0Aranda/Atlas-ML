import type { Dataset } from '../../types/ml';
import { dot, mean, norm, type Vector } from './linalg';

/** Keeps a covariance invertible when the data lies on a line. */
const COVARIANCE_RIDGE = 1e-6;
const BALL_SEGMENTS = 64;

/** The metrics algorithms can use on the 2D plot. */
export const PlaneMetric = { Euclidean: 0, Manhattan: 1, Chebyshev: 2 } as const;
export type PlaneMetric = (typeof PlaneMetric)[keyof typeof PlaneMetric];

export const PLANE_METRIC_OPTIONS = [
	{ label: 'Euclidean', value: PlaneMetric.Euclidean },
	{ label: 'Manhattan', value: PlaneMetric.Manhattan },
	{ label: 'Chebyshev', value: PlaneMetric.Chebyshev },
];

/** Generalizes the geometric distances: p = 1 is Manhattan and p = 2 is Euclidean. */
export function minkowski(a: Vector, b: Vector, p: number): number {
	return a.reduce((sum, v, i) => sum + Math.abs(v - b[i]) ** p, 0) ** (1 / p);
}

export const euclidean = (a: Vector, b: Vector): number => minkowski(a, b, 2);

export const manhattan = (a: Vector, b: Vector): number => minkowski(a, b, 1);

/** The limit of Minkowski distance as p grows: only the largest difference counts. */
export const chebyshev = (a: Vector, b: Vector): number => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

/** Written out per metric, because plots call this for every sample at every pixel. */
export function planeDistance(metric: PlaneMetric, ax: number, ay: number, bx: number, by: number): number {
	const dx = Math.abs(ax - bx);
	const dy = Math.abs(ay - by);
	if (metric === PlaneMetric.Manhattan) return dx + dy;
	if (metric === PlaneMetric.Chebyshev) return Math.max(dx, dy);
	return Math.sqrt(dx * dx + dy * dy);
}

/** Outline of all points at a given distance: a circle, a diamond or a square. */
export function metricBall(metric: PlaneMetric, x: number, y: number, radius: number): [number, number][] {
	return Array.from({ length: BALL_SEGMENTS + 1 }, (_, i) => {
		const angle = ((i % BALL_SEGMENTS) / BALL_SEGMENTS) * Math.PI * 2;
		const [u, v] = [Math.cos(angle), Math.sin(angle)];
		const scale = radius / planeDistance(metric, 0, 0, u, v);
		return [x + u * scale, y + v * scale];
	});
}

/** One minus the cosine of the angle. Not a true metric: it can break the triangle inequality. */
export function cosineDistance(a: Vector, b: Vector): number {
	const lengths = norm(a) * norm(b);
	return lengths === 0 ? 1 : 1 - dot(a, b) / lengths;
}

/** Inverse of the 2 by 2 sample covariance, for use with mahalanobis. */
export function inverseCovariance2d(data: Dataset): number[][] {
	const mx = mean(data.map((p) => p.x));
	const my = mean(data.map((p) => p.y));
	const moment = (f: (dx: number, dy: number) => number) => mean(data.map((p) => f(p.x - mx, p.y - my)));
	const xx = moment((dx) => dx * dx) + COVARIANCE_RIDGE;
	const yy = moment((_, dy) => dy * dy) + COVARIANCE_RIDGE;
	const xy = moment((dx, dy) => dx * dy);
	const determinant = xx * yy - xy * xy;
	return [
		[yy / determinant, -xy / determinant],
		[-xy / determinant, xx / determinant],
	];
}

/** Euclidean distance after rescaling space so the data's spread is equal in every direction. */
export function mahalanobis(a: Vector, b: Vector, inverseCovariance: number[][]): number {
	const difference = a.map((v, i) => v - b[i]);
	return Math.sqrt(Math.max(0, dot(difference, inverseCovariance.map((row) => dot(row, difference)))));
}

/** Positions that differ. Only defined for sequences of equal length. */
export function hamming(a: string, b: string): number | null {
	if (a.length !== b.length) return null;
	return [...a].filter((c, i) => c !== b[i]).length;
}

/** One minus the share of all elements that both sets contain. */
export function jaccardDistance<T>(a: Set<T>, b: Set<T>): number {
	const union = new Set([...a, ...b]).size;
	if (!union) return 0;
	return 1 - [...a].filter((v) => b.has(v)).length / union;
}

/** Entry [i][j] is the fewest edits that turn the first i characters of a into the first j of b. */
export function levenshteinTable(a: string, b: string): number[][] {
	const table = Array.from({ length: a.length + 1 }, (_, i) =>
		Array.from({ length: b.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)),
	);
	for (let i = 1; i <= a.length; i++) {
		for (let j = 1; j <= b.length; j++) {
			const substitution = table[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1);
			table[i][j] = Math.min(table[i - 1][j] + 1, table[i][j - 1] + 1, substitution);
		}
	}
	return table;
}

export const levenshtein = (a: string, b: string): number => levenshteinTable(a, b)[a.length][b.length];

/** Lowest total cost of aligning two sequences, where points may repeat to absorb shifts in time. */
export function dtw(a: number[], b: number[]): { distance: number; path: [number, number][] } {
	const cost = a.map(() => new Array<number>(b.length).fill(Infinity));
	const at = (i: number, j: number) => (i < 0 || j < 0 ? Infinity : cost[i][j]);
	a.forEach((value, i) => {
		b.forEach((other, j) => {
			const previous = i === 0 && j === 0 ? 0 : Math.min(at(i - 1, j - 1), at(i - 1, j), at(i, j - 1));
			cost[i][j] = Math.abs(value - other) + previous;
		});
	});

	let [i, j] = [a.length - 1, b.length - 1];
	const path: [number, number][] = [[i, j]];
	while (i > 0 || j > 0) {
		const options: [number, number][] = [[i - 1, j - 1], [i - 1, j], [i, j - 1]];
		[i, j] = options.reduce((best, option) => (at(...option) < at(...best) ? option : best));
		path.unshift([i, j]);
	}
	return { distance: cost[a.length - 1][b.length - 1], path };
}

/** Extra nats needed to encode p using a code built for q. Not symmetric, so not a metric. */
export function klDivergence(p: number[], q: number[]): number {
	return p.reduce((sum, value, i) => {
		if (value === 0) return sum;
		return q[i] === 0 ? Infinity : sum + value * Math.log(value / q[i]);
	}, 0);
}

/** Earth mover's distance between two histograms on the same evenly spaced bins. */
export function wasserstein(p: number[], q: number[], binWidth: number): number {
	let carried = 0;
	let work = 0;
	p.forEach((value, i) => {
		carried += value - q[i];
		work += Math.abs(carried) * binWidth;
	});
	return work;
}
