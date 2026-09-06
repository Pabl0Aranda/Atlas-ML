import { describe, expect, it } from 'vitest';
import { generateDataset } from '../ml/datasets';
import { cosineDistance, euclidean, inverseCovariance2d, mahalanobis, manhattan } from '../ml/distance';
import {
	allDistances,
	EXPLORER_METRICS,
	explorerLevels,
	explorerScene,
	measure,
	type ExplorerContext,
	type ExplorerMetric,
} from './distanceDemos';

const samples = generateDataset('ellipse');
const context: ExplorerContext = { p: 3, inverseCovariance: inverseCovariance2d(samples), samples };
const a = { x: -0.3, y: 0.2 };
const b = { x: 0.4, y: 0.5 };

describe('measure', () => {
	it('agrees with the distance library', () => {
		expect(measure('euclidean', a, b, context)).toBeCloseTo(euclidean([a.x, a.y], [b.x, b.y]));
		expect(measure('manhattan', a, b, context)).toBeCloseTo(manhattan([a.x, a.y], [b.x, b.y]));
		expect(measure('cosine', a, b, context)).toBeCloseTo(cosineDistance([a.x, a.y], [b.x, b.y]));
		expect(measure('mahalanobis', a, b, context)).toBeCloseTo(mahalanobis([a.x, a.y], [b.x, b.y], context.inverseCovariance));
	});

	it('uses the chosen p for Minkowski distance', () => {
		expect(measure('minkowski', a, b, { ...context, p: 2 })).toBeCloseTo(measure('euclidean', a, b, context));
		expect(measure('minkowski', a, b, { ...context, p: 1 })).toBeCloseTo(measure('manhattan', a, b, context));
	});

	it('reports every metric at once', () => {
		expect(allDistances(a, b, context)).toHaveLength(EXPLORER_METRICS.length);
	});
});

describe('explorerScene', () => {
	const metrics = EXPLORER_METRICS.map((m) => m.value);

	it.each(metrics)('draws %s level curves at their stated distance', (metric: ExplorerMetric) => {
		const scene = explorerScene(metric, a, b, context);
		const levels = explorerLevels(metric);
		const curves = (scene.lines ?? []).filter((l) => l.tone === 'muted');
		expect(curves.length).toBeGreaterThan(10);
		for (const line of curves) {
			for (const [x, y] of line.path) {
				// Cosine distance changes fastest near the origin, where a grid cell spans many angles.
				if (metric === 'cosine' && Math.hypot(x, y) < 0.2) continue;
				const nearest = Math.min(...levels.map((l) => Math.abs(measure(metric, a, { x, y }, context) - l)));
				expect(nearest).toBeLessThan(metric === 'mahalanobis' ? 0.2 : 0.05);
			}
		}
	});

	it('marks both points', () => {
		expect(explorerScene('euclidean', a, b, context).markers).toEqual([{ ...a, cls: 0 }, { ...b, cls: 1 }]);
	});

	it('shows the samples only for Mahalanobis, which depends on them', () => {
		expect(explorerScene('euclidean', a, b, context).points).toHaveLength(0);
		expect(explorerScene('mahalanobis', a, b, context).points).toHaveLength(samples.length);
	});

	it('walks a grid path for Manhattan distance', () => {
		const path = explorerScene('manhattan', a, b, context).lines!.find((l) => l.tone === 'ink')!.path;
		expect(path).toEqual([[a.x, a.y], [b.x, a.y], [b.x, b.y]]);
	});

	it('survives a point at the origin, where cosine is undefined', () => {
		const origin = { x: 0, y: 0 };
		expect(measure('cosine', origin, b, context)).toBe(1);
		expect(() => explorerScene('cosine', origin, b, context)).not.toThrow();
	});
});
