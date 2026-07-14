import type { Dataset } from '../../../types/ml';
import type { AlgorithmSpec, SceneLine } from '../../../types/viz';
import { mean } from '../../ml/linalg';
import { fitMinMax, fitStandard, flagOutliers, iqrBounds, standardDeviation, trainTestSplit } from '../../ml/preprocessing';
import { classPoints, fixed, percent } from '../scene';

interface Stage {
	stage: number;
}

const advance = (s: Stage): Stage => ({ stage: s.stage + 1 });

function fences(data: Dataset, k: number): SceneLine[] {
	const style = { tone: 'muted', dashed: true } as const;
	const vertical = iqrBounds(data.map((p) => p.x), k).map((x): SceneLine => ({ ...style, path: [[x, -1], [x, 1]] }));
	const horizontal = iqrBounds(data.map((p) => p.y), k).map((y): SceneLine => ({ ...style, path: [[-1, y], [1, y]] }));
	return [...vertical, ...horizontal];
}

export const dataCleaning: AlgorithmSpec<Stage> = {
	datasets: ['dirty', 'trend-outliers', 'ellipse'],
	params: [{ key: 'k', label: 'Fence multiplier k', value: 1.5, min: 0.5, max: 3, step: 0.1, live: true }],
	labeled: false,
	speed: 1,
	thumbnailSteps: 1,
	legend: () => ['Kept', 'Flagged outlier'],
	init: () => ({ stage: 0 }),
	step: advance,
	isDone: (s) => s.stage >= 2,
	scene(s, data, p) {
		const flags = flagOutliers(data, p.k);
		const lines = s.stage > 0 ? fences(data, p.k) : [];
		if (s.stage === 2) return { points: data.filter((_, i) => !flags[i]).map((d) => ({ x: d.x, y: d.y, cls: 0 })), lines };
		return { points: data.map((d, i) => ({ x: d.x, y: d.y, cls: s.stage > 0 && flags[i] ? 1 : 0 })), lines };
	},
	metrics(s, data, p) {
		const flagged = flagOutliers(data, p.k).filter(Boolean).length;
		return [
			{ label: 'Samples', value: String(s.stage === 2 ? data.length - flagged : data.length) },
			{ label: 'Flagged', value: s.stage > 0 ? String(flagged) : '–' },
		];
	},
	caption(s) {
		if (s.stage === 0) return 'The raw data. Step to compute the IQR fences on each axis.';
		if (s.stage === 1) return 'Points outside the dashed fences are flagged. Inspect them before deciding: an outlier may be a real event.';
		return 'Flagged points are removed. Note how many samples this cost.';
	},
};

const CANVAS_Z_SCALE = 0.3;

function scaled(data: Dataset, method: number): Dataset {
	const fit = method === 1 ? fitStandard : fitMinMax;
	const zoom = method === 1 ? CANVAS_Z_SCALE : 1;
	const sx = fit(data.map((p) => p.x));
	const sy = fit(data.map((p) => p.y));
	return data.map((p) => ({ ...p, x: sx(p.x) * zoom, y: sy(p.y) * zoom }));
}

export const featureScaling: AlgorithmSpec<Stage> = {
	datasets: ['wide', 'ellipse', 'dirty'],
	params: [
		{ key: 'method', label: 'Scaler', value: 1, live: true, options: [{ label: 'Min-max', value: 0 }, { label: 'Standardization', value: 1 }] },
	],
	labeled: false,
	speed: 1,
	thumbnailSteps: 0,
	legend: () => ['Samples'],
	init: () => ({ stage: 0 }),
	step: advance,
	isDone: (s) => s.stage >= 1,
	scene: (s, data, p) => ({ points: classPoints(s.stage ? scaled(data, p.method) : data).map((d) => ({ ...d, cls: 0 })) }),
	metrics(s, data, p) {
		const zoom = s.stage && p.method === 1 ? CANVAS_Z_SCALE : 1;
		const shown = s.stage ? scaled(data, p.method) : data;
		const xs = shown.map((d) => d.x / zoom);
		const ys = shown.map((d) => d.y / zoom);
		return [
			{ label: 'Mean x', value: fixed(mean(xs), 2) },
			{ label: 'Std x', value: fixed(standardDeviation(xs), 2) },
			{ label: 'Mean y', value: fixed(mean(ys), 2) },
			{ label: 'Std y', value: fixed(standardDeviation(ys), 2) },
		];
	},
	caption(s, _, p) {
		if (s.stage === 0) return 'Raw features: x spreads far more than y, so x would dominate any distance. Step to scale.';
		return p.method === 1
			? 'Each feature now has mean 0 and standard deviation 1. The plot is zoomed out so z-scores fit.'
			: 'Each feature now spans exactly 0 to 1, so the cloud fills the top-right quadrant.';
	},
};

interface SplitState extends Stage {
	isTest: boolean[];
}

const share = (data: Dataset, include: (i: number) => boolean) => {
	const selected = data.filter((_, i) => include(i));
	return selected.length ? selected.filter((d) => d.label === 1).length / selected.length : 0;
};

export const dataSplitting: AlgorithmSpec<SplitState> = {
	datasets: ['imbalanced', 'moons', 'outliers'],
	params: [
		{ key: 'ratio', label: 'Test share', value: 0.2, min: 0.1, max: 0.5, step: 0.05 },
		{ key: 'stratified', label: 'Sampling', value: 0, options: [{ label: 'Random', value: 0 }, { label: 'Stratified', value: 1 }] },
		{ key: 'seed', label: 'Shuffle seed', value: 1, min: 1, max: 20, step: 1 },
	],
	labeled: true,
	speed: 1,
	thumbnailSteps: 1,
	legend: () => ['Class A', 'Class B'],
	init: (data, p) => ({ stage: 0, isTest: trainTestSplit(data, p.ratio, p.stratified === 1, p.seed) }),
	step: (s) => ({ ...s, stage: 1 }),
	isDone: (s) => s.stage >= 1,
	scene: (s, data) => ({ points: data.map((d, i) => ({ x: d.x, y: d.y, cls: d.label, ring: s.stage > 0 && s.isTest[i] })) }),
	metrics(s, data) {
		const test = s.isTest.filter(Boolean).length;
		return [
			{ label: 'Train samples', value: s.stage ? String(data.length - test) : '–' },
			{ label: 'Test samples', value: s.stage ? String(test) : '–' },
			{ label: 'Class B in train', value: s.stage ? percent(share(data, (i) => !s.isTest[i])) : '–' },
			{ label: 'Class B in test', value: s.stage ? percent(share(data, (i) => s.isTest[i])) : '–' },
		];
	},
	caption: (s) =>
		s.stage === 0
			? 'Every sample is unassigned. Step to split.'
			: 'Ringed points form the test set, which the model must never see during training.',
};
