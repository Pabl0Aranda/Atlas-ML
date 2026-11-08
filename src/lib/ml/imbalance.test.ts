import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { classWeights, MINORITY, minorityScores, nearest, oversample, smote, undersample, weightedLogisticStep } from './imbalance';

const data = generateDataset('imbalanced', 120, 7);
const rare = data.filter((p) => p.label === MINORITY);
const common = data.filter((p) => p.label !== MINORITY);

describe('imbalanced data', () => {
	it('starts with one rare sample in ten', () => {
		expect(rare).toHaveLength(12);
		expect(common).toHaveLength(108);
	});

	it('undersamples the common class down to the size of the rare one', () => {
		const { kept, removed } = undersample(data);
		expect(kept.filter((p) => p.label === MINORITY)).toHaveLength(12);
		expect(kept.filter((p) => p.label !== MINORITY)).toHaveLength(12);
		expect(removed).toHaveLength(96);
		expect(removed.every((p) => p.label !== MINORITY)).toBe(true);
	});

	it('oversamples with exact copies of rare samples', () => {
		const copies = oversample(data);
		expect(copies).toHaveLength(96);
		expect(copies.every((c) => rare.some((r) => r.x === c.x && r.y === c.y && r.label === MINORITY))).toBe(true);
	});

	it('places each SMOTE sample between a rare sample and one of its k nearest rare neighbors', () => {
		const synthetic = smote(data, 3);
		expect(synthetic).toHaveLength(96);
		for (const { point, from, to } of synthetic) {
			expect(nearest(rare, from, 3)).toContain(to);
			const cross = (to.x - from.x) * (point.y - from.y) - (to.y - from.y) * (point.x - from.x);
			expect(cross).toBeCloseTo(0, 10);
			expect(Math.min(from.x, to.x) - 1e-12).toBeLessThanOrEqual(point.x);
			expect(Math.max(from.x, to.x) + 1e-12).toBeGreaterThanOrEqual(point.x);
			expect(point.label).toBe(MINORITY);
		}
		expect(smote(data.filter((p) => p.label !== MINORITY || p === rare[0]), 3)).toEqual([]);
	});

	it('gives each class the same total weight', () => {
		const [w0, w1] = classWeights(data);
		expect(w0 * 108).toBeCloseTo(w1 * 12);
		expect(w0 * 108 + w1 * 12).toBeCloseTo(120);
	});

	it('pulls the boundary toward the common class when the rare class weighs more', () => {
		let plain = [0, 0, 0];
		let weighted = [0, 0, 0];
		for (let i = 0; i < 200; i++) {
			plain = weightedLogisticStep(plain, data, [1, 1], 2);
			weighted = weightedLogisticStep(weighted, data, classWeights(data), 2);
		}
		// A larger bias means more of the plot is predicted rare.
		expect(weighted[0]).toBeGreaterThan(plain[0]);
	});

	it('scores accuracy, recall and precision on the rare class', () => {
		const test = [
			{ x: 0, y: 0, label: 1 },
			{ x: 0, y: 0, label: 1 },
			{ x: 1, y: 0, label: 0 },
			{ x: 1, y: 0, label: 0 },
		];
		// Predicts rare only at x = 0, and also wrongly at one common sample.
		let calls = 0;
		const scores = minorityScores((x) => (x === 0 || calls++ === 0 ? 1 : 0), test);
		expect(scores).toEqual({ accuracy: 0.75, recall: 1, precision: 2 / 3 });
		expect(minorityScores(() => 0, test)).toEqual({ accuracy: 0.5, recall: 0, precision: 0 });
	});
});
