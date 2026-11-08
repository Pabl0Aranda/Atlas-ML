import { describe, expect, it } from 'vitest';
import type { Hyperparameters } from '../../types/ml';
import { generateDataset } from '../ml/datasets';
import { defaultParams } from './scene';
import { imbalancedData as spec, METHODS, TEST_SET } from './specs/imbalance';

const data = generateDataset('imbalanced', 120, 7);

const finalMetrics = (method: number): Record<string, string> => {
	const params: Hyperparameters = { ...defaultParams(spec), method };
	let state = spec.init(data, params);
	while (!spec.isDone(state, params)) state = spec.step(state, data, params);
	return Object.fromEntries(spec.metrics(state, data, params).map((m) => [m.label, m.value]));
};

// The imbalanced data page and its preset notes quote these figures.
describe('imbalanced data presets', () => {
	it('tests on 400 samples that keep the real balance, 40 of them rare', () => {
		expect(TEST_SET).toHaveLength(400);
		expect(TEST_SET.filter((p) => p.label === 1)).toHaveLength(40);
	});

	it('without rebalancing, scores high accuracy but misses 9 of 40 rare samples', () => {
		expect(finalMetrics(0)).toMatchObject({ 'Test accuracy': '96.8%', 'Rare class recall': '77.5%', 'Rare class precision': '88.6%' });
	});

	it('with any rebalancing, catches 36 of 40 rare samples at a lower precision', () => {
		expect(finalMetrics(1)).toMatchObject({ 'Training: common / rare': '12 / 12', 'Rare class recall': '90.0%', 'Rare class precision': '73.5%' });
		for (const method of [2, 3, 4]) {
			expect(finalMetrics(method), METHODS[method]).toMatchObject({ 'Test accuracy': '95.5%', 'Rare class recall': '90.0%', 'Rare class precision': '72.0%' });
		}
		expect(finalMetrics(3)['Training: common / rare']).toBe('108 / 108');
	});
});
