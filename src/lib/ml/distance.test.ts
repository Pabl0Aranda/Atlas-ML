import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import {
	chebyshev,
	cosineDistance,
	dtw,
	euclidean,
	hamming,
	inverseCovariance2d,
	jaccardDistance,
	klDivergence,
	levenshtein,
	levenshteinTable,
	mahalanobis,
	manhattan,
	metricBall,
	minkowski,
	planeDistance,
	PlaneMetric,
	wasserstein,
} from './distance';

describe('geometric distances', () => {
	const a = [0, 0];
	const b = [3, 4];

	it('match a 3-4-5 triangle', () => {
		expect(euclidean(a, b)).toBeCloseTo(5);
		expect(manhattan(a, b)).toBeCloseTo(7);
		expect(chebyshev(a, b)).toBeCloseTo(4);
	});

	it('are the Minkowski family', () => {
		expect(minkowski(a, b, 1)).toBeCloseTo(manhattan(a, b));
		expect(minkowski(a, b, 2)).toBeCloseTo(euclidean(a, b));
		expect(minkowski(a, b, 60)).toBeCloseTo(chebyshev(a, b), 4);
	});

	it('shrink as p grows', () => {
		expect(minkowski(a, b, 3)).toBeLessThan(minkowski(a, b, 2));
		expect(minkowski(a, b, 3)).toBeGreaterThan(chebyshev(a, b));
	});

	it('measure the plane with each metric', () => {
		expect(planeDistance(PlaneMetric.Euclidean, 0, 0, 3, 4)).toBeCloseTo(5);
		expect(planeDistance(PlaneMetric.Manhattan, 0, 0, 3, 4)).toBeCloseTo(7);
		expect(planeDistance(PlaneMetric.Chebyshev, 0, 0, 3, 4)).toBeCloseTo(4);
	});

	it('trace balls whose outline sits at the radius', () => {
		for (const metric of Object.values(PlaneMetric)) {
			const ball = metricBall(metric, 0.2, -0.1, 0.3);
			expect(ball.at(0)).toEqual(ball.at(-1));
			for (const [x, y] of ball) expect(planeDistance(metric, 0.2, -0.1, x, y)).toBeCloseTo(0.3);
		}
	});
});

describe('cosine distance', () => {
	it('ignores length and depends only on direction', () => {
		expect(cosineDistance([1, 2], [2, 4])).toBeCloseTo(0);
		expect(cosineDistance([1, 0], [0, 5])).toBeCloseTo(1);
		expect(cosineDistance([1, 0], [-3, 0])).toBeCloseTo(2);
	});

	it('treats a zero vector as unrelated, not as an error', () => {
		expect(cosineDistance([0, 0], [1, 1])).toBe(1);
	});

	it('can break the triangle inequality, so it is not a true metric', () => {
		const [a, b, c] = [[1, 0], [1, 1], [0, 1]];
		expect(cosineDistance(a, c)).toBeGreaterThan(cosineDistance(a, b) + cosineDistance(b, c));
	});
});

describe('Mahalanobis distance', () => {
	it('equals Euclidean distance on uncorrelated unit-variance data', () => {
		expect(mahalanobis([0, 0], [3, 4], [[1, 0], [0, 1]])).toBeCloseTo(5);
	});

	it('discounts distance along the direction the data varies', () => {
		const inverse = inverseCovariance2d(generateDataset('ellipse'));
		const along = mahalanobis([0, 0], [0.3, 0.3], inverse);
		const across = mahalanobis([0, 0], [0.3, -0.3], inverse);
		expect(Math.abs(along - across)).toBeGreaterThan(0.1);
	});

	it('inverts the covariance of known data', () => {
		const data = [[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([x, y]) => ({ x, y, label: 0 }));
		const inverse = inverseCovariance2d(data);
		expect(inverse[0][0]).toBeCloseTo(1);
		expect(inverse[0][1]).toBeCloseTo(0);
	});

	it('stays finite on degenerate data', () => {
		const inverse = inverseCovariance2d([{ x: 0, y: 0, label: 0 }, { x: 1, y: 1, label: 0 }]);
		expect(inverse.flat().every(Number.isFinite)).toBe(true);
	});
});

describe('Hamming distance', () => {
	it('counts the positions that differ', () => {
		expect(hamming('karolin', 'kathrin')).toBe(3);
		expect(hamming('same', 'same')).toBe(0);
	});

	it('is undefined for different lengths', () => {
		expect(hamming('abc', 'ab')).toBeNull();
	});
});

describe('Jaccard distance', () => {
	it('compares overlap with the union', () => {
		expect(jaccardDistance(new Set([1, 2, 3]), new Set([2, 3, 4]))).toBeCloseTo(0.5);
		expect(jaccardDistance(new Set([1]), new Set([2]))).toBe(1);
	});

	it('treats two empty sets as identical', () => {
		expect(jaccardDistance(new Set(), new Set())).toBe(0);
	});
});

describe('Levenshtein distance', () => {
	it('counts the fewest edits', () => {
		expect(levenshtein('kitten', 'sitting')).toBe(3);
		expect(levenshtein('', 'abc')).toBe(3);
		expect(levenshtein('abc', 'abc')).toBe(0);
	});

	it('fills a table with one more row and column than the strings', () => {
		const table = levenshteinTable('ab', 'abc');
		expect(table).toHaveLength(3);
		expect(table[0]).toEqual([0, 1, 2, 3]);
		expect(table[2][3]).toBe(1);
	});
});

describe('dynamic time warping', () => {
	it('is zero for the same shape stretched in time', () => {
		expect(dtw([0, 1, 2], [0, 0, 1, 1, 2, 2]).distance).toBeCloseTo(0);
	});

	it('beats a point-by-point comparison on shifted signals', () => {
		const a = [0, 0, 1, 2, 1, 0, 0, 0];
		const b = [0, 0, 0, 0, 1, 2, 1, 0];
		const pointwise = a.reduce((sum, v, i) => sum + Math.abs(v - b[i]), 0);
		expect(dtw(a, b).distance).toBeLessThan(pointwise);
	});

	it('returns a path from start to end that never goes backward', () => {
		const { path } = dtw([1, 2, 3, 4], [1, 3, 4]);
		expect(path[0]).toEqual([0, 0]);
		expect(path.at(-1)).toEqual([3, 2]);
		for (let i = 1; i < path.length; i++) {
			expect(path[i][0]).toBeGreaterThanOrEqual(path[i - 1][0]);
			expect(path[i][1]).toBeGreaterThanOrEqual(path[i - 1][1]);
		}
	});
});

describe('distribution distances', () => {
	const p = [0.5, 0.5, 0, 0];
	const q = [0.25, 0.25, 0.25, 0.25];

	it('KL divergence is zero only for identical distributions', () => {
		expect(klDivergence(q, q)).toBeCloseTo(0);
		expect(klDivergence(p, q)).toBeGreaterThan(0);
	});

	it('KL divergence is not symmetric, so it is not a metric', () => {
		expect(klDivergence(p, q)).toBeCloseTo(Math.log(2));
		expect(klDivergence(q, p)).toBe(Infinity);
	});

	it('Wasserstein distance is the work to move mass', () => {
		expect(wasserstein([1, 0, 0], [0, 0, 1], 1)).toBeCloseTo(2);
		expect(wasserstein([1, 0, 0], [0, 1, 0], 0.5)).toBeCloseTo(0.5);
	});

	it('Wasserstein distance is symmetric and stays finite without overlap', () => {
		expect(wasserstein(p, q, 1)).toBeCloseTo(wasserstein(q, p, 1));
		expect(Number.isFinite(wasserstein([1, 0], [0, 1], 1))).toBe(true);
	});
});
