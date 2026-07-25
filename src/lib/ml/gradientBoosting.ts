import type { Dataset } from '../../types/ml';

/** Keeps the starting log-odds finite when every sample has the same class. */
const MIN_PROBABILITY = 1e-6;
/** Newton leaves explode when a leaf is already confident, so they are capped. */
const MAX_LEAF = 4;

type Feature = 'x' | 'y';

export type RegressionNode =
	| { kind: 'leaf'; value: number }
	| { kind: 'split'; feature: Feature; threshold: number; left: RegressionNode; right: RegressionNode };

export interface BoostingParams {
	learningRate: number;
	maxDepth: number;
}

export interface BoostingState {
	/** Log-odds of the positive class, before any tree. */
	base: number;
	trees: RegressionNode[];
	/** Current log-odds per sample. */
	scores: number[];
	loss: number;
	lossHistory: number[];
}

interface Row {
	x: number;
	y: number;
	/** Negative gradient of the log loss: label minus predicted probability. */
	residual: number;
	/** Second derivative of the log loss: p(1 - p). */
	hessian: number;
}

const sigmoid = (score: number) => 1 / (1 + Math.exp(-score));
const positive = (label: number) => (label > 0 ? 1 : 0);

function logLoss(scores: number[], data: Dataset): number {
	const total = scores.reduce((sum, score, i) => {
		const p = Math.min(1 - 1e-12, Math.max(1e-12, sigmoid(score)));
		return sum - (positive(data[i].label) ? Math.log(p) : Math.log(1 - p));
	}, 0);
	return total / scores.length;
}

export function initBoosting(data: Dataset): BoostingState {
	const share = data.reduce((sum, p) => sum + positive(p.label), 0) / data.length;
	const clipped = Math.min(1 - MIN_PROBABILITY, Math.max(MIN_PROBABILITY, share));
	const base = Math.log(clipped / (1 - clipped));
	const scores = data.map(() => base);
	const loss = logLoss(scores, data);
	return { base, trees: [], scores, loss, lossHistory: [loss] };
}

/** The split that most reduces squared error on the residuals. */
function bestResidualSplit(rows: Row[]): { feature: Feature; threshold: number } | null {
	let best: { feature: Feature; threshold: number } | null = null;
	let bestGain = -Infinity;
	const total = rows.reduce((sum, r) => sum + r.residual, 0);
	for (const feature of ['x', 'y'] as const) {
		const sorted = [...rows].sort((a, b) => a[feature] - b[feature]);
		let left = 0;
		for (let i = 1; i < sorted.length; i++) {
			left += sorted[i - 1].residual;
			if (sorted[i][feature] === sorted[i - 1][feature]) continue;
			const right = total - left;
			const gain = left ** 2 / i + right ** 2 / (sorted.length - i);
			if (gain > bestGain) {
				bestGain = gain;
				best = { feature, threshold: (sorted[i][feature] + sorted[i - 1][feature]) / 2 };
			}
		}
	}
	return best;
}

/** Leaves take one Newton step on the log loss, shrunk by the learning rate. */
function leafValue(rows: Row[], learningRate: number): number {
	const residual = rows.reduce((sum, r) => sum + r.residual, 0);
	const hessian = rows.reduce((sum, r) => sum + r.hessian, 0);
	const step = residual / Math.max(hessian, 1e-9);
	return learningRate * Math.min(MAX_LEAF, Math.max(-MAX_LEAF, step));
}

function fitTree(rows: Row[], depth: number, learningRate: number): RegressionNode {
	const split = depth > 0 && rows.length >= 2 ? bestResidualSplit(rows) : null;
	if (!split) return { kind: 'leaf', value: leafValue(rows, learningRate) };
	const { feature, threshold } = split;
	return {
		kind: 'split',
		feature,
		threshold,
		left: fitTree(rows.filter((r) => r[feature] <= threshold), depth - 1, learningRate),
		right: fitTree(rows.filter((r) => r[feature] > threshold), depth - 1, learningRate),
	};
}

function evaluate(node: RegressionNode, x: number, y: number): number {
	let current = node;
	while (current.kind === 'split') {
		current = (current.feature === 'x' ? x : y) <= current.threshold ? current.left : current.right;
	}
	return current.value;
}

/** Adds one small tree that corrects what the ensemble still gets wrong. */
export function boostingStep(state: BoostingState, data: Dataset, params: BoostingParams): BoostingState {
	const rows = data.map((p, i): Row => {
		const probability = sigmoid(state.scores[i]);
		return {
			x: p.x,
			y: p.y,
			residual: positive(p.label) - probability,
			hessian: probability * (1 - probability),
		};
	});
	const tree = fitTree(rows, params.maxDepth, params.learningRate);
	const scores = state.scores.map((score, i) => score + evaluate(tree, data[i].x, data[i].y));
	const loss = logLoss(scores, data);
	return { ...state, trees: [...state.trees, tree], scores, loss, lossHistory: [...state.lossHistory, loss] };
}

export function boostingProbability(state: BoostingState, x: number, y: number): number {
	return sigmoid(state.trees.reduce((score, tree) => score + evaluate(tree, x, y), state.base));
}

export const boostingClassify = (state: BoostingState, x: number, y: number): number =>
	boostingProbability(state, x, y) >= 0.5 ? 1 : 0;
