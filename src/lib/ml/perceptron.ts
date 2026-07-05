import type { Dataset } from '../../types/ml';

export interface PerceptronState {
	weights: [number, number];
	bias: number;
	epoch: number;
	mistakes: number;
	mistakeHistory: number[];
}

export const perceptronClassify = (s: PerceptronState, x: number, y: number): number =>
	s.weights[0] * x + s.weights[1] * y + s.bias > 0 ? 1 : 0;

export function initPerceptron(): PerceptronState {
	return { weights: [0.1, -0.1], bias: 0, epoch: 0, mistakes: -1, mistakeHistory: [] };
}

/** One pass of Rosenblatt's rule: nudge the boundary toward each misclassified point. */
export function perceptronEpoch(state: PerceptronState, data: Dataset, learningRate: number): PerceptronState {
	const weights: [number, number] = [...state.weights];
	let bias = state.bias;
	let mistakes = 0;
	for (const p of data) {
		const update = learningRate * (p.label - (weights[0] * p.x + weights[1] * p.y + bias > 0 ? 1 : 0));
		if (update !== 0) {
			weights[0] += update * p.x;
			weights[1] += update * p.y;
			bias += update;
			mistakes += 1;
		}
	}
	return {
		weights,
		bias,
		epoch: state.epoch + 1,
		mistakes,
		mistakeHistory: [...state.mistakeHistory, mistakes],
	};
}
