import { encode } from './dqn';
import { createRng, gaussian } from './random';
import { isTerminal, transition, type GridWorld } from './tabularRl';

const HIDDEN_UNITS = 32;
const ACTION_COUNT = 4;
const MINIBATCH_SIZE = 64;
const MAX_EPISODE_MOVES = 100;
/** Keeps value targets near 1 in magnitude. */
const REWARD_SCALE = 0.05;

/** One hidden tanh layer with a linear output. */
export interface Network {
	hidden: number[][];
	hiddenBias: number[];
	output: number[][];
	outputBias: number[];
}

export interface PpoParams {
	learningRate: number;
	gamma: number;
	/** GAE smoothing between one-step and full-episode advantage estimates. */
	lambda: number;
	/** How far the probability ratio may move before its gradient is cut off. */
	clip: number;
	entropyBonus: number;
	/** Moves collected between updates. */
	rollout: number;
	/** Passes over each rollout. */
	epochs: number;
}

interface Sample {
	cell: number;
	action: number;
	reward: number;
	/** Probability of the action under the policy that collected it. */
	oldProbability: number;
	value: number;
	next: number;
	terminal: boolean;
	/** True when the episode ended here, either by terminal cell or by time limit. */
	ended: boolean;
}

export interface PpoState {
	policy: Network;
	critic: Network;
	rollout: Sample[];
	position: number;
	totalMoves: number;
	updates: number;
	episode: number;
	moves: number;
	episodeReturn: number;
	returns: number[];
	/** Share of samples whose gradient the clip removed in the last update. */
	clippedShare: number;
	entropy: number;
	rng: () => number;
}

function createNetwork(inputs: number, outputs: number, outputScale: number, rng: () => number): Network {
	const layer = (rows: number, cols: number, scale: number) =>
		Array.from({ length: rows }, () => Array.from({ length: cols }, () => gaussian(rng) * scale));
	return {
		hidden: layer(HIDDEN_UNITS, inputs, Math.sqrt(1 / inputs)),
		hiddenBias: new Array(HIDDEN_UNITS).fill(0),
		output: layer(outputs, HIDDEN_UNITS, outputScale),
		outputBias: new Array(outputs).fill(0),
	};
}

function forward(net: Network, input: number[]): { activations: number[]; outputs: number[] } {
	const activations = net.hidden.map((row, j) => Math.tanh(row.reduce((s, w, i) => s + w * input[i], net.hiddenBias[j])));
	const outputs = net.output.map((row, k) => row.reduce((s, w, j) => s + w * activations[j], net.outputBias[k]));
	return { activations, outputs };
}

/** Applies one gradient step, given dLoss/dOutput for a single sample. */
function backward(net: Network, input: number[], activations: number[], outputGradient: number[], rate: number) {
	const hiddenGradient = activations.map((a, j) => {
		const back = outputGradient.reduce((s, g, k) => s + g * net.output[k][j], 0);
		return back * (1 - a * a);
	});
	outputGradient.forEach((g, k) => {
		activations.forEach((a, j) => (net.output[k][j] -= rate * g * a));
		net.outputBias[k] -= rate * g;
	});
	hiddenGradient.forEach((g, j) => {
		input.forEach((x, i) => {
			if (x) net.hidden[j][i] -= rate * g * x;
		});
		net.hiddenBias[j] -= rate * g;
	});
}

export function softmax(logits: number[]): number[] {
	const top = Math.max(...logits);
	const exps = logits.map((l) => Math.exp(l - top));
	const total = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / total);
}

export function actionProbabilities(policy: Network, world: GridWorld, cell: number): number[] {
	return softmax(forward(policy, encode(world, cell)).outputs);
}

export function stateValue(critic: Network, world: GridWorld, cell: number): number {
	return forward(critic, encode(world, cell)).outputs[0];
}

export function initPpo(world: GridWorld, seed = 23): PpoState {
	const rng = createRng(seed);
	const inputs = world.cols + world.rows;
	return {
		// A small output scale starts the policy close to uniform, so it explores.
		policy: createNetwork(inputs, ACTION_COUNT, 0.01, rng),
		critic: createNetwork(inputs, 1, Math.sqrt(1 / HIDDEN_UNITS), rng),
		rollout: [],
		position: world.start,
		totalMoves: 0,
		updates: 0,
		episode: 0,
		moves: 0,
		episodeReturn: 0,
		returns: [],
		clippedShare: 0,
		entropy: Math.log(ACTION_COUNT),
		rng,
	};
}

/**
 * Generalized advantage estimation: a discounted sum of TD errors, walked backward.
 * Time-limit endings still bootstrap from the next state, as only true terminals are worth zero.
 */
export function computeAdvantages(
	samples: Pick<Sample, 'reward' | 'value' | 'terminal' | 'ended'>[],
	nextValues: number[],
	gamma: number,
	lambda: number,
): number[] {
	const advantages = new Array(samples.length).fill(0);
	let running = 0;
	for (let t = samples.length - 1; t >= 0; t--) {
		const s = samples[t];
		const delta = s.reward + (s.terminal ? 0 : gamma * nextValues[t]) - s.value;
		running = delta + (s.ended ? 0 : gamma * lambda * running);
		advantages[t] = running;
	}
	return advantages;
}

function sampleAction(probabilities: number[], rng: () => number): number {
	let target = rng();
	for (let a = 0; a < probabilities.length - 1; a++) {
		target -= probabilities[a];
		if (target <= 0) return a;
	}
	return probabilities.length - 1;
}

/** The clipped surrogate update, run for several epochs over one rollout. */
function update(state: PpoState, world: GridWorld, params: PpoParams): { clippedShare: number; entropy: number } {
	const { policy, critic, rollout, rng } = state;
	const nextValues = rollout.map((s) => stateValue(critic, world, s.next));
	const advantages = computeAdvantages(rollout, nextValues, params.gamma, params.lambda);
	const targets = advantages.map((a, i) => a + rollout[i].value);

	const center = advantages.reduce((a, b) => a + b, 0) / advantages.length;
	const spread = Math.sqrt(advantages.reduce((s, a) => s + (a - center) ** 2, 0) / advantages.length) || 1;
	const normalized = advantages.map((a) => (a - center) / spread);

	let clipped = 0;
	let entropySum = 0;
	let visited = 0;
	const order = rollout.map((_, i) => i);
	for (let epoch = 0; epoch < params.epochs; epoch++) {
		for (let i = order.length - 1; i > 0; i--) {
			const j = Math.floor(rng() * (i + 1));
			[order[i], order[j]] = [order[j], order[i]];
		}
		const rate = params.learningRate / MINIBATCH_SIZE;
		for (const index of order) {
			const sample = rollout[index];
			const advantage = normalized[index];
			const input = encode(world, sample.cell);

			const { activations, outputs } = forward(policy, input);
			const probabilities = softmax(outputs);
			const ratio = probabilities[sample.action] / sample.oldProbability;
			const isClipped =
				(advantage > 0 && ratio > 1 + params.clip) || (advantage < 0 && ratio < 1 - params.clip);
			const entropy = -probabilities.reduce((s, p) => s + p * Math.log(p + 1e-12), 0);

			const gradient = probabilities.map((p, k) => {
				const surrogate = isClipped ? 0 : -advantage * ratio * ((k === sample.action ? 1 : 0) - p);
				const bonus = params.entropyBonus * p * (Math.log(p + 1e-12) + entropy);
				return surrogate + bonus;
			});
			backward(policy, input, activations, gradient, rate);

			const value = forward(critic, input);
			backward(critic, input, value.activations, [value.outputs[0] - targets[index]], rate);

			clipped += isClipped ? 1 : 0;
			entropySum += entropy;
			visited += 1;
		}
	}
	return { clippedShare: clipped / visited, entropy: entropySum / visited };
}

/** One agent move. A full rollout triggers a policy update and is then discarded. */
export function ppoStep(state: PpoState, world: GridWorld, params: PpoParams): PpoState {
	const cell = state.position;
	const input = encode(world, cell);
	const probabilities = softmax(forward(state.policy, input).outputs);
	const action = sampleAction(probabilities, state.rng);
	const { next, reward } = transition(world, cell, action);
	const terminal = isTerminal(world, next);
	const moves = state.moves + 1;
	const ended = terminal || moves >= MAX_EPISODE_MOVES;

	const rollout = [
		...state.rollout,
		{
			cell,
			action,
			reward: reward * REWARD_SCALE,
			oldProbability: probabilities[action],
			value: forward(state.critic, input).outputs[0],
			next,
			terminal,
			ended,
		},
	];

	const episodeReturn = state.episodeReturn + reward;
	const progressed: PpoState = ended
		? {
				...state,
				position: world.start,
				episode: state.episode + 1,
				moves: 0,
				episodeReturn: 0,
				returns: [...state.returns, episodeReturn],
			}
		: { ...state, position: next, moves, episodeReturn };
	const base = { ...progressed, rollout, totalMoves: state.totalMoves + 1 };

	if (rollout.length < params.rollout) return base;
	return { ...base, ...update(base, world, params), rollout: [], updates: state.updates + 1 };
}
