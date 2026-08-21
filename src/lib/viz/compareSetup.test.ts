import { describe, expect, it } from 'vitest';
import { compareData, decodeCompare, decodePair, defaultCompareSetup, encodeCompare, partnerFor, presetQuery, sharedDatasets, type CompareOption } from './compareSetup';
import { COMPARE_PRESETS } from './presets';
import { loadSpec } from './registry';
import type { AlgorithmSpec } from '../../types/viz';

const options: CompareOption[] = [
	{ id: 'knn', name: 'KNN', datasets: ['moons', 'circles', 'xor'] },
	{ id: 'svm', name: 'SVM', datasets: ['linear', 'moons', 'circles'] },
	{ id: 'linear-regression', name: 'Linear Regression', datasets: ['trend', 'curve'] },
	{ id: 'polynomial-regression', name: 'Polynomial Regression', datasets: ['curve', 'trend'] },
];

const specOf = (over: Partial<AlgorithmSpec>) =>
	({ datasets: ['moons', 'circles'], params: [{ key: 'k', label: 'k', value: 3, min: 1, max: 9 }], holdout: true, ...over }) as AlgorithmSpec;

const specA = specOf({});
const specB = specOf({ datasets: ['circles', 'moons', 'xor'], params: [{ key: 'C', label: 'C', value: 1, min: 0, max: 5 }] });

describe('pairing', () => {
	it('finds shared datasets in the first algorithm’s order', () => {
		expect(sharedDatasets(options[0], options[1])).toEqual(['moons', 'circles']);
		expect(sharedDatasets(options[0], options[2])).toEqual([]);
	});

	it('keeps a partner that shares data', () => {
		expect(partnerFor(options, 'knn', 'svm')).toBe('svm');
	});

	it('replaces a partner with nothing in common', () => {
		expect(partnerFor(options, 'knn', 'linear-regression')).toBe('svm');
		expect(partnerFor(options, 'linear-regression', 'knn')).toBe('polynomial-regression');
	});

	it('falls back to the algorithm itself when nothing else fits', () => {
		const lonely: CompareOption[] = [{ id: 'knn', name: 'KNN', datasets: ['moons'] }, { id: 'pca', name: 'PCA', datasets: ['ellipse'] }];
		expect(partnerFor(lonely, 'pca', 'knn')).toBe('pca');
	});

	it('starts with KNN against SVM', () => {
		expect(decodePair('', options)).toEqual({ a: 'knn', b: 'svm' });
	});

	it('picks a different partner when a link names only the first algorithm', () => {
		expect(decodePair('?a=svm', options)).toEqual({ a: 'svm', b: 'knn' });
	});

	it('allows the same algorithm on both sides when asked', () => {
		expect(decodePair('?a=svm&b=svm', options)).toEqual({ a: 'svm', b: 'svm' });
	});

	it('repairs links with unknown or incompatible algorithms', () => {
		expect(decodePair('?a=nope&b=nope', options)).toEqual({ a: 'knn', b: 'svm' });
		expect(decodePair('?a=knn&b=linear-regression', options)).toEqual({ a: 'knn', b: 'svm' });
	});
});

describe('compare links', () => {
	const pair = { a: 'knn', b: 'svm' } as const;

	it('uses the first shared dataset by default', () => {
		expect(defaultCompareSetup(specA, specB).dataset).toBe('moons');
	});

	it('round-trips a changed setup, with parameters per side', () => {
		const setup = { dataset: 'circles' as const, seed: 9, paramsA: { k: 7 }, paramsB: { C: 2 }, holdout: true };
		const query = encodeCompare(pair, setup, specA, specB);
		expect(decodePair(query, options)).toEqual(pair);
		expect(decodeCompare(query, specA, specB)).toEqual(setup);
	});

	it('keeps same-named parameters apart', () => {
		const same = specOf({});
		const setup = { ...defaultCompareSetup(same, same), paramsA: { k: 1 }, paramsB: { k: 9 } };
		const decoded = decodeCompare(encodeCompare({ a: 'knn', b: 'knn' }, setup, same, same), same, same);
		expect([decoded.paramsA.k, decoded.paramsB.k]).toEqual([1, 9]);
	});

	it('ignores a dataset only one side supports', () => {
		expect(decodeCompare('?dataset=xor', specA, specB).dataset).toBe('moons');
	});

	it('holds out only when both sides can be scored', () => {
		expect(decodeCompare('?holdout=1', specA, specB).holdout).toBe(true);
		expect(decodeCompare('?holdout=1', specA, specOf({ holdout: false })).holdout).toBe(false);
	});
});

describe('compareData', () => {
	it('gives both sides the same split', () => {
		const setup = { ...defaultCompareSetup(specA, specB), holdout: true };
		expect(compareData(setup)).toEqual(compareData(setup));
		expect(compareData(setup).test).toHaveLength(30);
	});
});

describe('guided pairs', () => {
	it.each(COMPARE_PRESETS)('$label is applied exactly as written', async (preset) => {
		const [a, b] = await Promise.all([loadSpec(preset.a), loadSpec(preset.b)]);
		const search = presetQuery(preset);
		const setup = decodeCompare(search, a, b);
		expect(setup.dataset).toBe(preset.dataset);
		expect(setup.holdout).toBe(preset.holdout ?? false);
		expect(setup.paramsA).toMatchObject(preset.paramsA ?? {});
		expect(setup.paramsB).toMatchObject(preset.paramsB ?? {});
	});
});
