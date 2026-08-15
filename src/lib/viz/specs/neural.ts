import type { AlgorithmSpec, Diagram, DiagramNode } from '../../../types/viz';
import { accuracy } from '../../ml/knn';
import { forward, initMlp, mlpProbability, mlpStep, type MlpState } from '../../ml/mlp';
import { initPerceptron, perceptronClassify, perceptronEpoch, type PerceptronState } from '../../ml/perceptron';
import { classPoints, fixed, percent } from '../scene';

const MAX_PERCEPTRON_EPOCHS = 300;
const MAX_MLP_EPOCHS = 5000;

export const perceptron: AlgorithmSpec<PerceptronState> = {
	datasets: ['linear', 'outliers', 'xor', 'circles'],
	params: [{ key: 'learningRate', label: 'Learning rate α', value: 0.1, min: 0.01, max: 1, step: 0.01 }],
	labeled: true,
	holdout: true,
	speed: 1,
	thumbnailSteps: 40,
	legend: () => ['Class A', 'Class B'],
	init: () => initPerceptron(),
	step: (s, data, p) => perceptronEpoch(s, data, p.learningRate),
	isDone: (s) => s.mistakes === 0 || s.epoch >= MAX_PERCEPTRON_EPOCHS,
	scene: (s, data) => ({ region: (x, y) => perceptronClassify(s, x, y), points: classPoints(data) }),
	metrics: (s, data) => [
		{ label: 'Epoch', value: String(s.epoch) },
		{ label: 'Mistakes last epoch', value: s.mistakes < 0 ? '–' : String(s.mistakes) },
		{ label: 'Accuracy', value: percent(accuracy(data, (x, y) => perceptronClassify(s, x, y))) },
	],
	caption(s) {
		if (s.epoch === 0) return 'The boundary starts with arbitrary small weights.';
		if (s.mistakes === 0) return `Converged in ${s.epoch} epochs: every sample is on the correct side.`;
		return `The rule corrected ${s.mistakes} mistakes this epoch. On non-separable data it never settles.`;
	},
	series: (s) => ({ label: 'Mistakes per epoch', values: s.mistakeHistory }),
};

function networkDiagram(s: MlpState, probe: { x: number; y: number }): Diagram {
	const activations = forward(s, probe.x, probe.y);
	const offsets: number[] = [];
	const nodes: DiagramNode[] = [];
	activations.forEach((layer, l) => {
		offsets.push(nodes.length);
		layer.forEach((value, i) => {
			nodes.push({
				x: l / (activations.length - 1),
				y: (i + 1) / (layer.length + 1),
				label: value.toFixed(2),
				cls: value >= (l === activations.length - 1 ? 0.5 : 0) ? 1 : 0,
				strength: l === activations.length - 1 ? Math.abs(value - 0.5) * 2 : Math.min(Math.abs(value), 1),
			});
		});
	});
	const edges = s.weights.flatMap((layer, l) =>
		layer.flatMap((row, j) => row.map((weight, i) => ({ from: offsets[l] + i, to: offsets[l + 1] + j, weight }))),
	);
	return { title: 'Activations at the cursor', nodes, edges };
}

const hiddenLayers = (p: Record<string, number>) => new Array(p.layers).fill(p.units);

export const mlp: AlgorithmSpec<MlpState> = {
	datasets: ['moons', 'circles', 'xor', 'linear', 'outliers'],
	params: [
		{ key: 'layers', label: 'Hidden layers', value: 2, min: 1, max: 3, step: 1 },
		{ key: 'units', label: 'Units per layer', value: 6, min: 2, max: 8, step: 1 },
		{ key: 'learningRate', label: 'Learning rate α', value: 0.5, min: 0.01, max: 2, step: 0.01 },
	],
	labeled: true,
	holdout: true,
	speed: 8,
	thumbnailSteps: 900,
	legend: () => ['Class A', 'Class B'],
	init: (data, p) => initMlp(data, hiddenLayers(p)),
	step: (s, data, p) => mlpStep(s, data, p.learningRate),
	isDone: (s) => s.epoch >= MAX_MLP_EPOCHS || !Number.isFinite(s.loss),
	scene: (s, data) => ({
		region: (x, y) => (mlpProbability(s, x, y) > 0.5 ? 1 : 0),
		points: classPoints(data),
	}),
	metrics: (s, data) => [
		{ label: 'Epoch', value: String(s.epoch) },
		{ label: 'Cross-entropy', value: fixed(s.loss) },
		{ label: 'Accuracy', value: percent(accuracy(data, (x, y) => (mlpProbability(s, x, y) > 0.5 ? 1 : 0))) },
	],
	caption: (s) =>
		s.epoch === 0
			? 'Random weights give a near-arbitrary boundary. Press Play to run backpropagation.'
			: `After ${s.epoch} epochs of backpropagation the cross-entropy is ${fixed(s.loss)}. Hover the plot to trace activations.`,
	series: (s) => ({ label: 'Loss per epoch', values: s.lossHistory }),
	diagram: networkDiagram,
};
