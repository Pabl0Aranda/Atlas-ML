import type { Dataset, Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec, SceneLine } from '../../../types/viz';
import { fitGp, gpPredict, type GpModel } from '../../ml/gaussianProcess';
import { ellipse, gmmStep, initGmm, mostLikely, density, type GmmState } from '../../ml/gmm';
import { accuracy } from '../../ml/knn';
import { fitLda, ldaScore } from '../../ml/lda';
import { initLogistic, logisticProbability, logisticStep, type LogisticState } from '../../ml/logisticRegression';
import { classifyNaiveBayes, fitNaiveBayes, type Feature } from '../../ml/naiveBayes';
import { predict } from '../../ml/regression';
import { initRegularized, regularizedStep, type RegularizedState } from '../../ml/regularizedRegression';
import { classPoints, contour, fixed, percent } from '../scene';

const CLASSES = ['Class A', 'Class B'];

interface Stage {
	stage: number;
}

const NB_STAGES: Feature[][] = [[], ['x'], ['x', 'y']];

export const naiveBayes: AlgorithmSpec<Stage> = {
	datasets: ['linear', 'circles', 'moons', 'xor', 'imbalanced'],
	params: [],
	labeled: true,
	speed: 1,
	thumbnailSteps: 2,
	legend: () => CLASSES,
	init: () => ({ stage: 0 }),
	step: (s) => ({ stage: s.stage + 1 }),
	isDone: (s) => s.stage >= 2,
	scene(s, data) {
		const model = fitNaiveBayes(data);
		return { region: (x, y) => classifyNaiveBayes(model, x, y, NB_STAGES[s.stage]), points: classPoints(data) };
	},
	metrics(s, data) {
		const model = fitNaiveBayes(data);
		return [
			{ label: 'Evidence used', value: NB_STAGES[s.stage].join(', ') || 'prior only' },
			{ label: 'Prior of class B', value: percent(model.find((c) => c.label === 1)?.prior ?? 0) },
			{ label: 'Accuracy', value: percent(accuracy(data, (x, y) => classifyNaiveBayes(model, x, y, NB_STAGES[s.stage]))) },
		];
	},
	caption: (s) =>
		[
			'With no evidence, the prior alone predicts the most common class everywhere. Step to add feature x.',
			'Feature x alone updates the belief, so the boundaries are vertical. Step to add feature y.',
			'Both features multiply in, as if they were independent. The boundary is curved, because each class has its own variances.',
		][s.stage],
};

export const logisticRegression: AlgorithmSpec<LogisticState> = {
	datasets: ['linear', 'outliers', 'circles', 'xor', 'moons'],
	params: [
		{ key: 'degree', label: 'Features', value: 1, options: [{ label: 'Linear', value: 1 }, { label: 'Quadratic', value: 2 }] },
		{ key: 'learningRate', label: 'Learning rate α', value: 1, min: 0.1, max: 5, step: 0.1 },
		{ key: 'l2', label: 'L2 regularizer λ', value: 0, min: 0, max: 0.2, step: 0.005 },
	],
	labeled: true,
	speed: 5,
	thumbnailSteps: 400,
	legend: () => CLASSES,
	init: (data, p) => initLogistic(data, p.degree, p.l2),
	step: (s, data, p) => logisticStep(s, data, p.degree, p.learningRate, p.l2),
	isDone: (s) => s.epoch >= 3000 || !Number.isFinite(s.loss),
	scene(s, data, p) {
		const f = (x: number, y: number) => logisticProbability(s.weights, x, y, p.degree);
		if (s.epoch === 0) return { points: classPoints(data) };
		return {
			region: (x, y) => (f(x, y) > 0.5 ? 1 : 0),
			points: classPoints(data),
			lines: [
				...contour(f, 0.5, { tone: 'ink' }),
				...contour(f, 0.9, { tone: 'muted', dashed: true }),
				...contour(f, 0.1, { tone: 'muted', dashed: true }),
			],
		};
	},
	metrics: (s, data, p) => [
		{ label: 'Epoch', value: String(s.epoch) },
		{ label: 'Log loss', value: fixed(s.loss) },
		{ label: 'Accuracy', value: percent(accuracy(data, (x, y) => (logisticProbability(s.weights, x, y, p.degree) > 0.5 ? 1 : 0))) },
	],
	caption: (s) =>
		s.epoch === 0
			? 'All weights start at zero, so every point gets probability 0.5. Press Play.'
			: 'The solid line is probability 0.5, and the dashes mark 0.1 and 0.9. Narrower dashes mean a more confident model.',
	series: (s) => ({ label: 'Log loss per epoch', values: s.lossHistory }),
};

export const lda: AlgorithmSpec<Stage> = {
	datasets: ['linear', 'outliers', 'imbalanced', 'circles'],
	params: [],
	labeled: true,
	speed: 1,
	thumbnailSteps: 2,
	legend: () => CLASSES,
	init: () => ({ stage: 0 }),
	step: (s) => ({ stage: s.stage + 1 }),
	isDone: (s) => s.stage >= 2,
	scene(s, data) {
		const m = fitLda(data);
		const markers = m.means.map((c, cls) => ({ ...c, cls }));
		if (s.stage === 0) return { points: classPoints(data), markers };
		const length = Math.hypot(...m.direction) || 1;
		const [u, v] = [m.direction[0] / length, m.direction[1] / length];
		const mid = { x: (m.means[0].x + m.means[1].x) / 2, y: (m.means[0].y + m.means[1].y) / 2 };
		const axis: SceneLine = { tone: 'ink', dashed: true, path: [[mid.x - 3 * u, mid.y - 3 * v], [mid.x + 3 * u, mid.y + 3 * v]] };
		if (s.stage === 1) return { points: classPoints(data), markers, lines: [axis] };
		const f = (x: number, y: number) => ldaScore(m, x, y);
		return {
			region: (x, y) => (f(x, y) > 0 ? 1 : 0),
			points: classPoints(data),
			markers,
			lines: [axis, ...contour(f, 0, { tone: 'ink' })],
		};
	},
	metrics(s, data) {
		const m = fitLda(data);
		return [
			{ label: 'Fisher ratio', value: fixed(m.fisherRatio, 2) },
			{ label: 'Accuracy', value: s.stage === 2 ? percent(accuracy(data, (x, y) => (ldaScore(m, x, y) > 0 ? 1 : 0))) : '–' },
		];
	},
	caption: (s) =>
		[
			'Large markers show each class mean. Step to find the discriminant direction.',
			'The dashed axis best separates the classes once projected. It accounts for spread, so it need not join the means.',
			'The boundary is perpendicular to that axis in the shared covariance geometry, and shifted by the class priors.',
		][s.stage],
};

export const gaussianMixture: AlgorithmSpec<GmmState> = {
	datasets: ['blobs', 'ellipse', 'moons', 'circles'],
	params: [{ key: 'k', label: 'Components k', value: 4, min: 1, max: 4, step: 1 }],
	labeled: false,
	speed: 1,
	thumbnailSteps: 30,
	legend: (p) => Array.from({ length: p.k }, (_, i) => `Component ${i + 1}`),
	init: (data, p) => initGmm(data, p.k),
	step: (s, data) => gmmStep(s, data),
	isDone: (s) => s.converged || s.iteration >= 200,
	scene: (s, data) => ({
		region: (x, y) => mostLikely(s.components.map((c) => c.weight * density(c, x, y))),
		points: data.map((p, i) => ({ x: p.x, y: p.y, cls: mostLikely(s.responsibilities[i]) })),
		markers: s.components.map((c, cls) => ({ x: c.x, y: c.y, cls })),
		lines: s.components.map((c): SceneLine => ({ tone: 'ink', path: ellipse(c) })),
	}),
	metrics(s) {
		const unsure = s.responsibilities.filter((r) => Math.max(...r) < 0.9).length;
		return [
			{ label: 'EM iteration', value: String(s.iteration) },
			{ label: 'Log-likelihood', value: fixed(s.logLikelihood, 2) },
			{ label: 'Uncertain points', value: String(unsure) },
		];
	},
	caption(s) {
		if (s.iteration === 0) return 'Round Gaussians start at k-means++ positions. Each ellipse marks two standard deviations.';
		if (s.converged) return `EM converged after ${s.iteration} iterations.`;
		return 'E-step: points share themselves between components. M-step: each Gaussian refits to its share.';
	},
	series: (s) => ({ label: 'Log-likelihood per iteration', values: s.history }),
};

function createPenaltySpec(l1Ratio: number | null): AlgorithmSpec<RegularizedState> {
	const penalty = (p: Hyperparameters) => ({ lambda: p.lambda, l1Ratio: l1Ratio ?? p.l1Ratio });
	const curve = (s: RegularizedState): SceneLine => ({
		tone: 'ink',
		path: Array.from({ length: 81 }, (_, i) => {
			const x = -1 + i / 40;
			return [x, Math.max(-3, Math.min(3, predict(s.weights, x)))];
		}),
	});
	return {
		datasets: ['curve', 'trend', 'trend-outliers'],
		params: [
			{ key: 'degree', label: 'Polynomial degree', value: 9, min: 1, max: 12, step: 1 },
			{ key: 'lambda', label: 'Penalty λ', value: 0.01, min: 0, max: 0.2, step: 0.002 },
			...(l1Ratio === null ? [{ key: 'l1Ratio', label: 'L1 ratio', value: 0.5, min: 0, max: 1, step: 0.05 }] : []),
		],
		labeled: false,
		speed: 10,
		thumbnailSteps: 800,
		legend: () => ['Samples'],
		init: (data, p) => initRegularized(data, p.degree, penalty(p)),
		step: (s, data, p) => regularizedStep(s, data, penalty(p)),
		isDone: (s) => s.change < 1e-6 || s.pass >= 3000,
		scene: (s, data) => ({ points: data.map((d) => ({ x: d.x, y: d.y, cls: 0 })), lines: [curve(s)] }),
		metrics(s) {
			const terms = s.weights.slice(1);
			return [
				{ label: 'Pass', value: String(s.pass) },
				{ label: 'Objective', value: fixed(s.loss, 4) },
				{ label: 'Non-zero terms', value: `${terms.filter((w) => w !== 0).length} of ${terms.length}` },
				{ label: 'Largest weight', value: fixed(Math.max(...terms.map(Math.abs)), 2) },
			];
		},
		caption: (s) =>
			s.pass === 0
				? 'Every coefficient starts at zero. Each pass solves them one at a time.'
				: s.change < 1e-6
					? `Converged after ${s.pass} passes.`
					: 'Coordinate descent updates one coefficient at a time, holding the rest fixed.',
		series: (s) => ({ label: 'Objective per pass', values: s.lossHistory }),
	};
}

export const ridge = createPenaltySpec(0);
export const lasso = createPenaltySpec(1);
export const elasticNet = createPenaltySpec(null);

const MAX_OBSERVED = 30;

interface GpState {
	count: number;
}

function gpModel(s: GpState, data: Dataset, p: Hyperparameters): GpModel {
	const seen = data.slice(0, s.count);
	return fitGp(seen.map((d) => d.x), seen.map((d) => d.y), { lengthScale: p.lengthScale, noise: p.noise });
}

const GRID = Array.from({ length: 101 }, (_, i) => -1 + i / 50);

export const gaussianProcess: AlgorithmSpec<GpState> = {
	datasets: ['curve', 'trend', 'trend-outliers'],
	editable: false,
	params: [
		{ key: 'lengthScale', label: 'Length scale ℓ', value: 0.3, min: 0.05, max: 1, step: 0.05, live: true },
		{ key: 'noise', label: 'Noise σ', value: 0.1, min: 0.01, max: 0.5, step: 0.01, live: true },
	],
	labeled: false,
	speed: 1,
	thumbnailSteps: 12,
	legend: () => ['Observed samples'],
	init: () => ({ count: 0 }),
	step: (s) => ({ count: s.count + 1 }),
	isDone: (s) => s.count >= MAX_OBSERVED,
	scene(s, data, p) {
		const model = gpModel(s, data, p);
		const predictions = GRID.map((x) => ({ x, ...gpPredict(model, x) }));
		const band = (sign: number): SceneLine => ({
			tone: 'muted',
			dashed: true,
			path: predictions.map((q) => [q.x, q.mean + sign * 2 * Math.sqrt(q.variance)]),
		});
		return {
			points: data.slice(0, s.count).map((d) => ({ x: d.x, y: d.y, cls: 0 })),
			lines: [band(1), band(-1), { tone: 'ink', path: predictions.map((q) => [q.x, q.mean]) }],
		};
	},
	metrics(s, data, p) {
		const model = gpModel(s, data, p);
		const average = GRID.reduce((sum, x) => sum + 2 * Math.sqrt(gpPredict(model, x).variance), 0) / GRID.length;
		return [
			{ label: 'Observed', value: `${s.count} of ${MAX_OBSERVED}` },
			{ label: 'Average ±2σ width', value: fixed(average, 3) },
		];
	},
	caption: (s) =>
		s.count === 0
			? 'Before any data, the prior is flat with wide uncertainty. Step to reveal one observation.'
			: 'The solid line is the posterior mean, and dashes mark about 95% of the function’s uncertainty. It narrows near data.',
};
