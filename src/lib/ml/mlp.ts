import type { Dataset } from '../../types/ml';
import { createRng, gaussian } from './random';

/** Fully connected network: 2 inputs, tanh hidden layers, one sigmoid output. */
export interface MlpState {
	/** weights[l][j][i] connects unit i of layer l to unit j of layer l+1. */
	weights: number[][][];
	biases: number[][];
	epoch: number;
	loss: number;
	lossHistory: number[];
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Returns every layer's activations, from the input to the output. */
export function forward(state: Pick<MlpState, 'weights' | 'biases'>, x: number, y: number): number[][] {
	const activations = [[x, y]];
	state.weights.forEach((layer, l) => {
		const isOutput = l === state.weights.length - 1;
		const input = activations[l];
		activations.push(
			layer.map((row, j) => {
				const z = row.reduce((s, w, i) => s + w * input[i], state.biases[l][j]);
				return isOutput ? sigmoid(z) : Math.tanh(z);
			}),
		);
	});
	return activations;
}

export const mlpProbability = (s: Pick<MlpState, 'weights' | 'biases'>, x: number, y: number): number =>
	forward(s, x, y).at(-1)![0];

/** Mean binary cross-entropy. */
export function mlpLoss(state: Pick<MlpState, 'weights' | 'biases'>, data: Dataset): number {
	const eps = 1e-9;
	const total = data.reduce((s, p) => {
		const q = mlpProbability(state, p.x, p.y);
		return s - (p.label ? Math.log(q + eps) : Math.log(1 - q + eps));
	}, 0);
	return total / Math.max(data.length, 1);
}

export function initMlp(data: Dataset, hidden: number[], seed = 5): MlpState {
	const rng = createRng(seed);
	const sizes = [2, ...hidden, 1];
	const weights = sizes.slice(1).map((size, l) => {
		const scale = Math.sqrt(1 / sizes[l]); // Xavier-style scaling keeps tanh out of saturation.
		return Array.from({ length: size }, () => Array.from({ length: sizes[l] }, () => gaussian(rng) * scale));
	});
	const biases = sizes.slice(1).map((size) => new Array(size).fill(0));
	const loss = mlpLoss({ weights, biases }, data);
	return { weights, biases, epoch: 0, loss, lossHistory: [loss] };
}

/** One full-batch backpropagation update. */
export function mlpStep(state: MlpState, data: Dataset, learningRate: number): MlpState {
	const gradW = state.weights.map((layer) => layer.map((row) => row.map(() => 0)));
	const gradB = state.biases.map((layer) => layer.map(() => 0));

	for (const p of data) {
		const a = forward(state, p.x, p.y);
		// Sigmoid + cross-entropy collapses the output delta to (prediction - target).
		let delta = [a.at(-1)![0] - p.label];
		for (let l = state.weights.length - 1; l >= 0; l--) {
			delta.forEach((d, j) => {
				gradB[l][j] += d;
				a[l].forEach((value, i) => (gradW[l][j][i] += d * value));
			});
			if (l > 0) {
				delta = a[l].map((value, i) => {
					const back = delta.reduce((s, d, j) => s + d * state.weights[l][j][i], 0);
					return back * (1 - value * value);
				});
			}
		}
	}

	const scale = learningRate / Math.max(data.length, 1);
	const weights = state.weights.map((layer, l) =>
		layer.map((row, j) => row.map((w, i) => w - scale * gradW[l][j][i])),
	);
	const biases = state.biases.map((layer, l) => layer.map((b, j) => b - scale * gradB[l][j]));
	const loss = mlpLoss({ weights, biases }, data);
	return { weights, biases, epoch: state.epoch + 1, loss, lossHistory: [...state.lossHistory, loss] };
}
