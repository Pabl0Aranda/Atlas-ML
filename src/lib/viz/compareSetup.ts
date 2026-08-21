import type { Dataset, DatasetKind, Hyperparameters } from '../../types/ml';
import type { AlgorithmId, AlgorithmSpec } from '../../types/viz';
import { composeData, decodeParams, decodeSeed, DEFAULT_SEED, encodeParams } from './sandboxSetup';
import { defaultParams } from './scene';

const DEFAULT_PAIR = { a: 'knn', b: 'svm' } as const;
const PREFIX_A = 'a_';
const PREFIX_B = 'b_';

/** What the page knows about an algorithm before its spec has been downloaded. */
export interface CompareOption {
	id: AlgorithmId;
	name: string;
	datasets: DatasetKind[];
}

export interface ComparePair {
	a: AlgorithmId;
	b: AlgorithmId;
}

export interface CompareSetup {
	dataset: DatasetKind;
	seed: number;
	paramsA: Hyperparameters;
	paramsB: Hyperparameters;
	holdout: boolean;
}

/** A pairing worth seeing, opened from the guided list. */
export interface ComparePreset extends ComparePair {
	label: string;
	/** What to look for. Figures come from running this pairing on the default seed. */
	note: string;
	dataset: DatasetKind;
	paramsA?: Hyperparameters;
	paramsB?: Hyperparameters;
	holdout?: boolean;
}

export const sharedDatasets = (first: { datasets: DatasetKind[] }, second: { datasets: DatasetKind[] }): DatasetKind[] =>
	first.datasets.filter((d) => second.datasets.includes(d));

/** The partner for a, keeping b if the two can run on the same data. */
export function partnerFor(options: CompareOption[], a: AlgorithmId, b: AlgorithmId): AlgorithmId {
	const first = options.find((o) => o.id === a)!;
	const current = options.find((o) => o.id === b);
	if (current && sharedDatasets(first, current).length) return b;
	return options.find((o) => o.id !== a && sharedDatasets(first, o).length)?.id ?? a;
}

/** Reads the two algorithms from a query string, repairing any pair that cannot run. */
export function decodePair(search: string, options: CompareOption[]): ComparePair {
	const query = new URLSearchParams(search);
	const known = (key: string) => options.find((o) => o.id === query.get(key))?.id;
	const a = known('a');
	if (!a) return DEFAULT_PAIR;
	return { a, b: partnerFor(options, a, known('b') ?? (a === DEFAULT_PAIR.a ? DEFAULT_PAIR.b : DEFAULT_PAIR.a)) };
}

export const defaultCompareSetup = (specA: AlgorithmSpec, specB: AlgorithmSpec): CompareSetup => ({
	dataset: sharedDatasets(specA, specB)[0],
	seed: DEFAULT_SEED,
	paramsA: defaultParams(specA),
	paramsB: defaultParams(specB),
	holdout: false,
});

export function encodeCompare({ a, b }: ComparePair, setup: CompareSetup, specA: AlgorithmSpec, specB: AlgorithmSpec): string {
	const base = defaultCompareSetup(specA, specB);
	const query = new URLSearchParams({ a, b });
	if (setup.dataset !== base.dataset) query.set('dataset', setup.dataset);
	if (setup.seed !== base.seed) query.set('seed', String(setup.seed));
	encodeParams(query, setup.paramsA, specA, PREFIX_A);
	encodeParams(query, setup.paramsB, specB, PREFIX_B);
	if (setup.holdout) query.set('holdout', '1');
	return query.toString();
}

export function decodeCompare(search: string, specA: AlgorithmSpec, specB: AlgorithmSpec): CompareSetup {
	const query = new URLSearchParams(search);
	const shared = sharedDatasets(specA, specB);
	return {
		dataset: shared.find((d) => d === query.get('dataset')) ?? shared[0],
		seed: decodeSeed(query),
		paramsA: decodeParams(query, specA, PREFIX_A),
		paramsB: decodeParams(query, specB, PREFIX_B),
		holdout: Boolean(specA.holdout && specB.holdout) && query.get('holdout') === '1',
	};
}

/** The one dataset, and one split, that both sides run on. */
export const compareData = ({ dataset, seed, holdout }: CompareSetup): { train: Dataset; test: Dataset } =>
	composeData({ dataset, seed, holdout, params: {}, added: [] });

/** Query string that opens a guided pairing. It needs no specs, so it works before they load. */
export function presetQuery({ a, b, dataset, paramsA, paramsB, holdout }: ComparePreset): string {
	const query = new URLSearchParams({ a, b, dataset });
	for (const [key, value] of Object.entries(paramsA ?? {})) query.set(PREFIX_A + key, String(value));
	for (const [key, value] of Object.entries(paramsB ?? {})) query.set(PREFIX_B + key, String(value));
	if (holdout) query.set('holdout', '1');
	return query.toString();
}
