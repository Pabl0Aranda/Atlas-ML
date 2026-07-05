import { createRng, gaussian } from './random';
import { greedyAction, isTerminal, transition, type GridWorld } from './tabularRl';

const HIDDEN_UNITS = 48;
const ACTION_COUNT = 4;
const BUFFER_SIZE = 5000;
const BATCH_SIZE = 32;
const WARMUP_MOVES = 200;
const MAX_EPISODE_MOVES = 200;
/** Keeps targets near 1 in magnitude, which plain SGD needs to stay stable. */
export const REWARD_SCALE = 0.05;

export interface QNetwork {
	/** hidden[j][i] connects input i to hidden unit j. */
	hidden: number[][];
	hiddenBias: number[];
	/** output[a][j] connects hidden unit j to the value of action a. */
	output: number[][];
	outputBias: number[];
}

interface Transition {
	cell: number;
	action: number;
	reward: number;
	next: number;
	terminal: boolean;
}

export interface DqnParams {
	learningRate: number;
	gamma: number;
	/** Exploration rate reached at the end of the decay. */
	epsilonEnd: number;
	/** Moves over which exploration decays from 1 to epsilonEnd. */
	epsilonDecay: number;
	/** Moves between copies of the online network into the target network. */
	targetSync: number;
}

export interface DqnState {
	online: QNetwork;
	target: QNetwork;
	/** Ring buffer, mutated in place: copying it on every move would dominate the runtime. */
	buffer: Transition[];
	position: number;
	totalMoves: number;
	episode: number;
	moves: number;
	episodeReturn: number;
	returns: number[];
	loss: number;
	rng: () => number;
}

/** Encodes a cell as one-hot column plus one-hot row, so nearby cells share inputs. */
export function encode(world: GridWorld, cell: number): number[] {
	const features = new Array(world.cols + world.rows).fill(0);
	features[cell % world.cols] = 1;
	features[world.cols + Math.floor(cell / world.cols)] = 1;
	return features;
}

function forward(net: QNetwork, input: number[]): { activations: number[]; values: number[] } {
	const activations = net.hidden.map((row, j) => {
		const sum = row.reduce((s, w, i) => s + w * input[i], net.hiddenBias[j]);
		return Math.max(0, sum);
	});
	const values = net.output.map((row, a) => row.reduce((s, w, j) => s + w * activations[j], net.outputBias[a]));
	return { activations, values };
}

export function qValues(net: QNetwork, world: GridWorld, cell: number): number[] {
	return forward(net, encode(world, cell)).values;
}

/** The network's values for every cell, in the same shape as a tabular Q-table. */
export function qTable(net: QNetwork, world: GridWorld): number[][] {
	return world.tiles.map((_, cell) => qValues(net, world, cell));
}

export function initDqn(world: GridWorld, seed = 17): DqnState {
	const rng = createRng(seed);
	const inputs = world.cols + world.rows;
	const layer = (rows: number, cols: number, scale: number) =>
		Array.from({ length: rows }, () => Array.from({ length: cols }, () => gaussian(rng) * scale));
	const online: QNetwork = {
		hidden: layer(HIDDEN_UNITS, inputs, Math.sqrt(2 / inputs)),
		hiddenBias: new Array(HIDDEN_UNITS).fill(0),
		output: layer(ACTION_COUNT, HIDDEN_UNITS, Math.sqrt(1 / HIDDEN_UNITS)),
		outputBias: new Array(ACTION_COUNT).fill(0),
	};
	return {
		online,
		target: structuredClone(online),
		buffer: [],
		position: world.start,
		totalMoves: 0,
		episode: 0,
		moves: 0,
		episodeReturn: 0,
		returns: [],
		loss: 0,
		rng,
	};
}

export function currentEpsilon(state: DqnState, params: DqnParams): number {
	const progress = Math.min(state.totalMoves / params.epsilonDecay, 1);
	return 1 + (params.epsilonEnd - 1) * progress;
}

/** One SGD step on a random minibatch of past transitions. Returns the mean squared TD error. */
function trainBatch(state: DqnState, world: GridWorld, params: DqnParams): number {
	const { online, target, buffer, rng } = state;
	const scale = params.learningRate / BATCH_SIZE;
	let squaredError = 0;
	for (let n = 0; n < BATCH_SIZE; n++) {
		const t = buffer[Math.floor(rng() * buffer.length)];
		// The frozen target network supplies the bootstrap value.
		const bootstrap = t.terminal ? 0 : Math.max(...qValues(target, world, t.next));
		const goal = t.reward + params.gamma * bootstrap;

		const input = encode(world, t.cell);
		const { activations, values } = forward(online, input);
		const error = values[t.action] - goal;
		squaredError += error * error;

		// Only the action actually taken receives a gradient.
		const outputRow = online.output[t.action];
		activations.forEach((value, j) => {
			if (value > 0) {
				const back = error * outputRow[j];
				const hiddenRow = online.hidden[j];
				input.forEach((x, i) => {
					if (x) hiddenRow[i] -= scale * back;
				});
				online.hiddenBias[j] -= scale * back;
			}
			outputRow[j] -= scale * error * value;
		});
		online.outputBias[t.action] -= scale * error;
	}
	return squaredError / BATCH_SIZE;
}

/** One agent move, stored in the replay buffer, followed by one training batch. */
export function dqnStep(state: DqnState, world: GridWorld, params: DqnParams): DqnState {
	const cell = state.position;
	const explore = state.rng() < currentEpsilon(state, params);
	const action = explore
		? Math.floor(state.rng() * ACTION_COUNT)
		: greedyAction(qValues(state.online, world, cell));
	const { next, reward } = transition(world, cell, action);
	const terminal = isTerminal(world, next);

	const totalMoves = state.totalMoves + 1;
	const memory = { cell, action, reward: reward * REWARD_SCALE, next, terminal };
	if (state.buffer.length < BUFFER_SIZE) state.buffer.push(memory);
	else state.buffer[totalMoves % BUFFER_SIZE] = memory;

	const loss = totalMoves >= WARMUP_MOVES ? trainBatch(state, world, params) : state.loss;
	const target = totalMoves % params.targetSync === 0 ? structuredClone(state.online) : state.target;

	const moves = state.moves + 1;
	const episodeReturn = state.episodeReturn + reward;
	const base = { ...state, target, totalMoves, loss };
	if (terminal || moves >= MAX_EPISODE_MOVES) {
		return {
			...base,
			position: world.start,
			episode: state.episode + 1,
			moves: 0,
			episodeReturn: 0,
			returns: [...state.returns, episodeReturn],
		};
	}
	return { ...base, position: next, moves, episodeReturn };
}
