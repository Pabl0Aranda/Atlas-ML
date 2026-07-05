import type { Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec } from '../../../types/viz';
import { accuracy } from '../../ml/knn';
import { initSvm, supportVectorCount, svmDecision, svmStep, type SvmParams, type SvmState } from '../../ml/svm';
import { contour, percent } from '../scene';

const MAX_PASSES = 200;

const toParams = (p: Hyperparameters): SvmParams => ({
	C: p.C,
	gamma: p.gamma,
	kernel: p.kernel === 1 ? 'rbf' : 'linear',
});

export const svm: AlgorithmSpec<SvmState> = {
	datasets: ['linear', 'outliers', 'circles', 'moons', 'xor'],
	params: [
		{ key: 'kernel', label: 'Kernel', value: 0, options: [{ label: 'Linear', value: 0 }, { label: 'RBF', value: 1 }] },
		{ key: 'C', label: 'Penalty C', value: 1, min: 0.1, max: 20, step: 0.1 },
		{ key: 'gamma', label: 'RBF width γ', value: 5, min: 0.5, max: 30, step: 0.5 },
	],
	labeled: true,
	speed: 1,
	thumbnailSteps: 30,
	legend: () => ['Class A (−1)', 'Class B (+1)'],
	init: (data) => initSvm(data),
	step: (s, data, p) => svmStep(s, data, toParams(p)),
	isDone: (s) => s.pass >= MAX_PASSES || (s.pass > 0 && s.changed === 0),
	scene(s, data, p) {
		const params = toParams(p);
		const f = (x: number, y: number) => svmDecision(s, data, params, x, y);
		if (s.pass === 0) return { points: data.map((d) => ({ x: d.x, y: d.y, cls: d.label })) };
		return {
			region: (x, y) => (f(x, y) > 0 ? 1 : 0),
			points: data.map((d, i) => ({ x: d.x, y: d.y, cls: d.label, ring: s.alphas[i] > 1e-6 })),
			lines: [
				...contour(f, 0, { tone: 'ink' }),
				...contour(f, 1, { tone: 'muted', dashed: true }),
				...contour(f, -1, { tone: 'muted', dashed: true }),
			],
		};
	},
	metrics(s, data, p) {
		const params = toParams(p);
		return [
			{ label: 'SMO pass', value: String(s.pass) },
			{ label: 'Support vectors', value: String(supportVectorCount(s)) },
			{ label: 'Accuracy', value: percent(accuracy(data, (x, y) => (svmDecision(s, data, params, x, y) > 0 ? 1 : 0))) },
		];
	},
	caption(s) {
		if (s.pass === 0) return 'All dual variables start at zero. Step to run one SMO pass over the samples.';
		if (s.changed === 0) return 'No pair of dual variables changed in this pass, so SMO has converged. Ringed points are support vectors.';
		return `This pass updated ${s.changed} pairs of dual variables. Ringed points are support vectors; dashes mark the margins.`;
	},
};
