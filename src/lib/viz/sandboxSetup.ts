import { generateDataset } from '../ml/datasets';
import { accuracy } from '../ml/knn';
import { trainTestSplit } from '../ml/preprocessing';
import type { Classifier, Dataset, DatasetKind, Hyperparameters } from '../../types/ml';
import type { AlgorithmSpec, Metric, ParamSpec, Scene } from '../../types/viz';
import { defaultParams, percent } from './scene';

const SAMPLES = 120;
export const DEFAULT_SEED = 7;
const MAX_SEED = 1_000_000;
const TEST_SHARE = 0.25;
const MAX_ADDED = 200;
/** Keeps links short. Three decimals are finer than a pixel on the plot. */
const ADDED_DIGITS = 3;
/** Separates parameters from the link's own keys, since a parameter can be named seed. */
const PARAM_PREFIX = 'p_';

/** Everything needed to rebuild a run from scratch. */
export interface SandboxSetup {
	dataset?: DatasetKind;
	seed: number;
	params: Hyperparameters;
	/** Keeps part of the data out of training, to score the model honestly. */
	holdout: boolean;
	/** Samples placed by hand on top of the generated dataset. */
	added: Dataset;
}

export const defaultSetup = (spec: AlgorithmSpec): SandboxSetup => ({
	dataset: spec.datasets[0],
	seed: DEFAULT_SEED,
	params: defaultParams(spec),
	holdout: false,
	added: [],
});

/** Builds the training data, plus the held-out samples the model never sees. */
export function composeData({ dataset, seed, holdout, added }: SandboxSetup): { train: Dataset; test: Dataset } {
	const generated = dataset ? generateDataset(dataset, SAMPLES, seed) : [];
	if (!holdout) return { train: [...generated, ...added], test: [] };
	const isTest = trainTestSplit(generated, TEST_SHARE, true, seed);
	return {
		train: [...generated.filter((_, i) => !isTest[i]), ...added],
		test: generated.filter((_, i) => isTest[i]),
	};
}

export function testAccuracy(region: Classifier | undefined, test: Dataset): number | null {
	return region && test.length ? accuracy(test, region) : null;
}

/** The Test accuracy metric, when samples are held out. */
export function heldOutMetrics(region: Classifier | undefined, test: Dataset): Metric[] {
	if (!test.length) return [];
	const score = testAccuracy(region, test);
	return [{ label: 'Test accuracy', value: score === null ? '–' : percent(score) }];
}

/** Adds held-out samples to a scene as hollow points. */
export function withHeldOut(scene: Scene, test: Dataset): Scene {
	if (!test.length) return scene;
	return { ...scene, points: [...scene.points, ...test.map((p) => ({ x: p.x, y: p.y, cls: p.label, hollow: true }))] };
}

const round = (value: number) => Number(value.toFixed(ADDED_DIGITS));

/** Adds every parameter that differs from its default. */
export function encodeParams(query: URLSearchParams, params: Hyperparameters, spec: AlgorithmSpec, prefix: string) {
	for (const { key, value } of spec.params) {
		if (params[key] !== value) query.set(prefix + key, String(params[key]));
	}
}

/** Query string for a setup. Default values are left out, so a plain page has a plain URL. */
export function encodeSetup(setup: SandboxSetup, spec: AlgorithmSpec): string {
	const base = defaultSetup(spec);
	const query = new URLSearchParams();
	if (setup.dataset !== base.dataset) query.set('dataset', setup.dataset!);
	if (setup.seed !== base.seed) query.set('seed', String(setup.seed));
	encodeParams(query, setup.params, spec, PARAM_PREFIX);
	if (setup.holdout) query.set('holdout', '1');
	if (setup.added.length) {
		query.set('added', setup.added.map((p) => `${round(p.x)},${round(p.y)},${p.label}`).join(';'));
	}
	return query.toString();
}

function decodeParam(param: ParamSpec, raw: string | null): number {
	const value = raw === null || raw === '' ? NaN : Number(raw);
	if (!Number.isFinite(value)) return param.value;
	if (param.options) return param.options.some((o) => o.value === value) ? value : param.value;
	return Math.min(param.max ?? value, Math.max(param.min ?? value, value));
}

export const decodeParams = (query: URLSearchParams, spec: AlgorithmSpec, prefix: string): Hyperparameters =>
	Object.fromEntries(spec.params.map((p) => [p.key, decodeParam(p, query.get(prefix + p.key))]));

export function decodeSeed(query: URLSearchParams): number {
	const seed = Number(query.get('seed'));
	return query.has('seed') && Number.isInteger(seed) && seed >= 0 && seed <= MAX_SEED ? seed : DEFAULT_SEED;
}

function decodeAdded(raw: string | null, spec: AlgorithmSpec, params: Hyperparameters): Dataset {
	if (!raw || spec.editable === false) return [];
	const classes = spec.labeled ? spec.legend(params).length : 1;
	return raw
		.split(';')
		.slice(0, MAX_ADDED)
		.flatMap((entry) => {
			const [x, y, label] = entry.split(',').map(Number);
			const valid = Math.abs(x) <= 1 && Math.abs(y) <= 1 && Number.isInteger(label) && label >= 0 && label < classes;
			return valid ? [{ x, y, label }] : [];
		});
}

/** Reads a setup from a query string. Links can be edited by hand, so nothing is trusted. */
export function decodeSetup(search: string, spec: AlgorithmSpec): SandboxSetup {
	const query = new URLSearchParams(search);
	const base = defaultSetup(spec);
	const dataset = spec.datasets.find((d) => d === query.get('dataset')) ?? base.dataset;
	const params = decodeParams(query, spec, PARAM_PREFIX);
	return {
		dataset,
		seed: decodeSeed(query),
		params,
		holdout: Boolean(spec.holdout) && query.get('holdout') === '1',
		added: decodeAdded(query.get('added'), spec, params),
	};
}
