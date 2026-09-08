import type { Dataset } from '../../types/ml';
import type { Metric, Scene, SceneLine } from '../../types/viz';
import { chebyshev, cosineDistance, dtw, euclidean, mahalanobis, manhattan, minkowski } from '../ml/distance';
import type { Point2D } from './interaction';
import { contour } from './scene';

export type ExplorerMetric = 'euclidean' | 'manhattan' | 'chebyshev' | 'minkowski' | 'cosine' | 'mahalanobis';

export const EXPLORER_METRICS: { value: ExplorerMetric; label: string; note: string }[] = [
	{ value: 'euclidean', label: 'Euclidean', note: 'Straight-line distance. Points at equal distance form circles.' },
	{ value: 'manhattan', label: 'Manhattan', note: 'Distance along a grid, like city blocks. Points at equal distance form diamonds.' },
	{ value: 'chebyshev', label: 'Chebyshev', note: 'Only the largest difference counts, like a chess king’s moves. Points at equal distance form squares.' },
	{ value: 'minkowski', label: 'Minkowski', note: 'One formula for the whole family. Move p from 1 towards 8 and the diamond swells through a circle towards a square.' },
	{ value: 'cosine', label: 'Cosine', note: 'Compares direction from the origin and ignores length, so equal distance forms rays instead of rings.' },
	{ value: 'mahalanobis', label: 'Mahalanobis', note: 'Measures in units of the data’s own spread, so equal distance forms ellipses shaped like the grey samples.' },
];

const LEVELS: Partial<Record<ExplorerMetric, number[]>> = {
	cosine: [0.1, 0.5, 1],
	mahalanobis: [1, 2, 3],
};
const DEFAULT_LEVELS = [0.25, 0.5, 0.75];

export interface ExplorerContext {
	/** Exponent for Minkowski distance. */
	p: number;
	inverseCovariance: number[][];
	/** The data behind inverseCovariance. */
	samples: Dataset;
}

export const explorerLevels = (metric: ExplorerMetric): number[] => LEVELS[metric] ?? DEFAULT_LEVELS;

export function measure(metric: ExplorerMetric, a: Point2D, b: Point2D, { p, inverseCovariance }: ExplorerContext): number {
	const [u, v] = [[a.x, a.y], [b.x, b.y]];
	switch (metric) {
		case 'euclidean':
			return euclidean(u, v);
		case 'manhattan':
			return manhattan(u, v);
		case 'chebyshev':
			return chebyshev(u, v);
		case 'minkowski':
			return minkowski(u, v, p);
		case 'cosine':
			return cosineDistance(u, v);
		case 'mahalanobis':
			return mahalanobis(u, v, inverseCovariance);
	}
}

export const allDistances = (a: Point2D, b: Point2D, context: ExplorerContext): Metric[] =>
	EXPLORER_METRICS.map(({ value, label }) => ({
		label: value === 'minkowski' ? `Minkowski (p = ${context.p})` : label,
		value: measure(value, a, b, context).toFixed(3),
	}));

/** How each metric travels between the two points. */
function connection(metric: ExplorerMetric, a: Point2D, b: Point2D): SceneLine[] {
	const style = { tone: 'ink', dashed: true } as const;
	if (metric === 'cosine') return [{ ...style, path: [[0, 0], [a.x, a.y]] }, { ...style, path: [[0, 0], [b.x, b.y]] }];
	if (metric === 'manhattan') return [{ ...style, path: [[a.x, a.y], [b.x, a.y], [b.x, b.y]] }];
	return [{ ...style, path: [[a.x, a.y], [b.x, b.y]] }];
}

/** Level curves around point a, plus a line to point b. */
export function explorerScene(metric: ExplorerMetric, a: Point2D, b: Point2D, context: ExplorerContext): Scene {
	const distanceFromA = (x: number, y: number) => measure(metric, a, { x, y }, context);
	return {
		points: metric === 'mahalanobis' ? context.samples.map((p) => ({ x: p.x, y: p.y, cls: 0, muted: true })) : [],
		lines: [
			...explorerLevels(metric).flatMap((level) => contour(distanceFromA, level, { tone: 'muted' })),
			...connection(metric, a, b),
		],
		markers: [{ ...a, cls: 0 }, { ...b, cls: 1 }],
	};
}

/** The distinct words in a text, ignoring case and punctuation. */
export const wordSet = (text: string): Set<string> => new Set(text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);

const SERIES_LENGTH = 48;
const BUMP_CENTER = 15;
const BUMP_WIDTH = 4;
const ALIGNMENT_STRIDE = 4;

const bump = (center: number) =>
	Array.from({ length: SERIES_LENGTH }, (_, i) => Math.exp(-((i - center) ** 2) / (2 * BUMP_WIDTH ** 2)));

/** Compares a signal with a time-shifted copy, point by point and with dynamic time warping. */
export function timeSeriesComparison(shift: number): { pointwise: number; warped: number; scene: Scene } {
	const first = bump(BUMP_CENTER);
	const second = bump(BUMP_CENTER + shift);
	const { distance, path } = dtw(first, second);
	const x = (i: number) => -0.95 + (1.9 * i) / (SERIES_LENGTH - 1);
	const top = (i: number) => 0.2 + 0.65 * first[i];
	const bottom = (i: number) => -0.85 + 0.65 * second[i];
	const curve = (y: (i: number) => number): SceneLine => ({ tone: 'ink', path: first.map((_, i) => [x(i), y(i)]) });
	const alignment = path
		.filter((_, n) => n % ALIGNMENT_STRIDE === 0)
		.map(([i, j]): SceneLine => ({ tone: 'muted', path: [[x(i), top(i)], [x(j), bottom(j)]] }));
	return {
		pointwise: first.reduce((sum, value, i) => sum + Math.abs(value - second[i]), 0),
		warped: distance,
		scene: { points: [], lines: [...alignment, curve(top), curve(bottom)] },
	};
}

const BINS = 81;
const BIN_WIDTH = 2 / (BINS - 1);

/** A Gaussian as a histogram on shared bins from -1 to 1. */
export function distributionBins(mean: number, sd: number): { values: number[]; width: number } {
	const raw = Array.from({ length: BINS }, (_, i) => Math.exp(-((-1 + i * BIN_WIDTH - mean) ** 2) / (2 * sd ** 2)));
	const total = raw.reduce((sum, v) => sum + v, 0);
	return { values: raw.map((v) => v / total), width: BIN_WIDTH };
}

export function distributionScene(p: number[], q: number[]): Scene {
	const peak = Math.max(...p, ...q);
	const curve = (values: number[], dashed: boolean): SceneLine => ({
		tone: 'ink',
		dashed,
		path: values.map((v, i) => [-1 + i * BIN_WIDTH, -0.9 + (1.7 * v) / peak]),
	});
	return { points: [], lines: [curve(p, false), curve(q, true)] };
}
