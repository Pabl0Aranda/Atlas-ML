import type { Dataset, Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec, SceneLine } from '../../../types/viz';
import { initRegression, predict, regressionStep, type RegressionState } from '../../ml/regression';
import { fixed } from '../scene';

const MAX_EPOCHS = 4000;
const clampY = (v: number) => Math.max(-3, Math.min(3, Number.isFinite(v) ? v : 3));

const shared: Omit<AlgorithmSpec<RegressionState>, 'datasets' | 'params' | 'thumbnailSteps' | 'init' | 'step'> = {
	labeled: false,
	speed: 4,
	legend: () => ['Samples'],
	isDone: (s) => s.epoch >= MAX_EPOCHS || !Number.isFinite(s.loss),
	scene(s, data) {
		const curve: SceneLine = {
			tone: 'ink',
			path: Array.from({ length: 81 }, (_, i) => {
				const x = -1 + i / 40;
				return [x, clampY(predict(s.weights, x))] as [number, number];
			}),
		};
		const residuals: SceneLine[] = data.map((p) => ({
			tone: 'muted',
			path: [[p.x, p.y], [p.x, clampY(predict(s.weights, p.x))]],
		}));
		return { points: data.map((p) => ({ x: p.x, y: p.y, cls: 0 })), lines: [...residuals, curve] };
	},
	metrics: (s) => [
		{ label: 'Epoch', value: String(s.epoch) },
		{ label: 'Loss', value: fixed(s.loss, 4) },
		{ label: 'Bias w₀', value: fixed(s.weights[0]) },
		{ label: 'Slope w₁', value: fixed(s.weights[1]) },
	],
	caption(s) {
		if (!Number.isFinite(s.loss)) return 'The learning rate is too high: each update overshoots and the loss explodes.';
		if (s.epoch === 0) return 'All weights start at zero. Thin lines show each residual.';
		return `After ${s.epoch} gradient steps the mean squared error is ${fixed(s.loss, 4)}.`;
	},
	series: (s) => ({ label: 'Loss per epoch', values: s.lossHistory }),
};

const train = (s: RegressionState, data: Dataset, p: Hyperparameters) => regressionStep(s, data, p.learningRate, p.l2 ?? 0);

export const linearRegression: AlgorithmSpec<RegressionState> = {
	...shared,
	datasets: ['trend', 'trend-outliers', 'curve'],
	params: [{ key: 'learningRate', label: 'Learning rate α', value: 0.1, min: 0.01, max: 1.2, step: 0.01 }],
	thumbnailSteps: 150,
	init: (data) => initRegression(data, 1),
	step: train,
};

export const polynomialRegression: AlgorithmSpec<RegressionState> = {
	...shared,
	datasets: ['curve', 'trend', 'trend-outliers'],
	params: [
		{ key: 'degree', label: 'Degree', value: 5, min: 1, max: 12, step: 1 },
		{ key: 'learningRate', label: 'Learning rate α', value: 0.3, min: 0.01, max: 1.2, step: 0.01 },
		{ key: 'l2', label: 'L2 regularizer λ', value: 0, min: 0, max: 0.2, step: 0.005 },
	],
	speed: 12,
	thumbnailSteps: 1500,
	init: (data, p) => initRegression(data, p.degree, p.l2),
	step: train,
};
