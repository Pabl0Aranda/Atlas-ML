import { describe, expect, it } from 'vitest';
import { generateDataset } from '../ml/datasets';
import { composeData, decodeSetup, defaultSetup, encodeSetup } from './sandboxSetup';
import type { AlgorithmSpec } from '../../types/viz';

const spec = {
	datasets: ['moons', 'circles'],
	params: [
		{ key: 'k', label: 'k', value: 3, min: 1, max: 9, step: 2 },
		{ key: 'seed', label: 'Shuffle seed', value: 1, min: 1, max: 20, step: 1 },
		{ key: 'kernel', label: 'Kernel', value: 0, options: [{ label: 'A', value: 0 }, { label: 'B', value: 1 }] },
	],
	labeled: true,
	legend: () => ['Class A', 'Class B'],
} as unknown as AlgorithmSpec;

const unlabeled = { ...spec, labeled: false } as AlgorithmSpec;
const fixedPlot = { ...spec, editable: false, datasets: [] } as unknown as AlgorithmSpec;

describe('share links', () => {
	it('encodes nothing for the default setup', () => {
		expect(encodeSetup(defaultSetup(spec), spec)).toBe('');
	});

	it('round-trips a changed setup', () => {
		const setup = {
			dataset: 'circles' as const,
			seed: 12,
			params: { k: 7, seed: 4, kernel: 1 },
			added: [{ x: 0.125, y: -0.5, label: 1 }],
		};
		expect(decodeSetup(encodeSetup(setup, spec), spec)).toEqual(setup);
	});

	it('keeps a parameter named seed apart from the dataset seed', () => {
		const setup = { ...defaultSetup(spec), seed: 30, params: { k: 3, seed: 2, kernel: 0 } };
		const decoded = decodeSetup(encodeSetup(setup, spec), spec);
		expect(decoded.seed).toBe(30);
		expect(decoded.params.seed).toBe(2);
	});

	it('ignores values the spec does not allow', () => {
		const decoded = decodeSetup('?dataset=nope&seed=abc&p_k=NaN&p_kernel=5&p_other=1', spec);
		expect(decoded).toEqual(defaultSetup(spec));
	});

	it('clamps slider values into range', () => {
		expect(decodeSetup('?p_k=500', spec).params.k).toBe(9);
		expect(decodeSetup('?p_k=-500', spec).params.k).toBe(1);
	});

	it('drops malformed samples, and limits their number', () => {
		expect(decodeSetup('?added=0.1,0.2,1;bad;5,0,0;0,0,9', spec).added).toEqual([{ x: 0.1, y: 0.2, label: 1 }]);
		const many = Array.from({ length: 900 }, () => '0,0,0').join(';');
		expect(decodeSetup(`?added=${many}`, spec).added.length).toBeLessThanOrEqual(200);
	});

	it('forces the class to 0 on unlabeled specs', () => {
		expect(decodeSetup('?added=0.1,0.2,1', unlabeled).added).toEqual([]);
		expect(decodeSetup('?added=0.1,0.2,0', unlabeled).added).toEqual([{ x: 0.1, y: 0.2, label: 0 }]);
	});

	it('ignores samples on specs that do not support them', () => {
		expect(decodeSetup('?added=0.1,0.2,0', fixedPlot).added).toEqual([]);
	});

	it('handles specs without datasets', () => {
		expect(defaultSetup(fixedPlot).dataset).toBeUndefined();
		expect(composeData(defaultSetup(fixedPlot))).toEqual([]);
	});
});

describe('composeData', () => {
	it('regenerates the dataset from its seed', () => {
		expect(composeData(defaultSetup(spec))).toEqual(generateDataset('moons', 120, 7));
	});

	it('adds samples placed by hand', () => {
		const added = [{ x: 0.5, y: 0.5, label: 1 }];
		expect(composeData({ ...defaultSetup(spec), added }).at(-1)).toEqual(added[0]);
	});
});
