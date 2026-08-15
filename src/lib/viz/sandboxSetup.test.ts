import { describe, expect, it } from 'vitest';
import { generateDataset } from '../ml/datasets';
import { composeData, decodeSetup, defaultSetup, encodeSetup, testAccuracy } from './sandboxSetup';
import type { AlgorithmSpec } from '../../types/viz';

const spec = {
	datasets: ['moons', 'circles'],
	params: [
		{ key: 'k', label: 'k', value: 3, min: 1, max: 9, step: 2 },
		{ key: 'seed', label: 'Shuffle seed', value: 1, min: 1, max: 20, step: 1 },
		{ key: 'kernel', label: 'Kernel', value: 0, options: [{ label: 'A', value: 0 }, { label: 'B', value: 1 }] },
	],
	labeled: true,
	holdout: true,
	legend: () => ['Class A', 'Class B'],
} as unknown as AlgorithmSpec;

const unlabeled = { ...spec, labeled: false, holdout: false } as AlgorithmSpec;
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
			holdout: true,
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
		const decoded = decodeSetup('?dataset=nope&seed=abc&p_k=NaN&p_kernel=5&p_other=1&holdout=maybe', spec);
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

	it('ignores holdout and samples on specs that do not support them', () => {
		expect(decodeSetup('?holdout=1', unlabeled).holdout).toBe(false);
		expect(decodeSetup('?added=0.1,0.2,0', fixedPlot).added).toEqual([]);
	});

	it('handles specs without datasets', () => {
		expect(defaultSetup(fixedPlot).dataset).toBeUndefined();
		expect(composeData(defaultSetup(fixedPlot))).toEqual({ train: [], test: [] });
	});
});

describe('composeData', () => {
	it('uses every generated sample when nothing is held out', () => {
		const { train, test } = composeData(defaultSetup(spec));
		expect(train).toEqual(generateDataset('moons', 120, 7));
		expect(test).toEqual([]);
	});

	it('holds out a quarter, keeping the class balance', () => {
		const { train, test } = composeData({ ...defaultSetup(spec), holdout: true });
		expect(test).toHaveLength(30);
		expect(train).toHaveLength(90);
		expect(test.filter((p) => p.label === 0)).toHaveLength(15);
	});

	it('regenerates the same split from the same setup', () => {
		const setup = { ...defaultSetup(spec), holdout: true };
		expect(composeData(setup)).toEqual(composeData(setup));
	});

	it('always trains on added samples', () => {
		const added = [{ x: 0.5, y: 0.5, label: 1 }];
		const { train, test } = composeData({ ...defaultSetup(spec), holdout: true, added });
		expect(train.at(-1)).toEqual(added[0]);
		expect(test).not.toContainEqual(added[0]);
	});
});

describe('testAccuracy', () => {
	it('scores the classifier on the held-out samples', () => {
		const test = [{ x: -1, y: 0, label: 0 }, { x: 1, y: 0, label: 1 }, { x: 1, y: 1, label: 0 }];
		expect(testAccuracy((x) => (x > 0 ? 1 : 0), test)).toBeCloseTo(2 / 3);
	});

	it('has no score without a classifier or samples', () => {
		expect(testAccuracy(undefined, [{ x: 0, y: 0, label: 0 }])).toBeNull();
		expect(testAccuracy(() => 0, [])).toBeNull();
	});
});
