import { createRng } from './random';

export type Tile = 'free' | 'wall' | 'cliff' | 'start' | 'goal';
export type RlAlgorithm = 'q-learning' | 'sarsa';

export interface GridWorld {
	cols: number;
	rows: number;
	tiles: Tile[];
	start: number;
	goal: number;
}

export interface RlParams {
	algorithm: RlAlgorithm;
	/** Learning rate. */
	alpha: number;
	/** Discount factor. */
	gamma: number;
	/** Exploration rate of the epsilon-greedy policy. */
	epsilon: number;
}

export interface RlState {
	/** q[cell][action], with actions ordered up, right, down, left. */
	q: number[][];
	position: number;
	/** The action already committed to for the next move. */
	action: number | null;
	episode: number;
	moves: number;
	episodeReturn: number;
	returns: number[];
	lastUpdate: { reward: number; target: number } | null;
	rng: () => number;
}

const ACTIONS: [number, number][] = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const STEP_REWARD = -1;
const CLIFF_REWARD = -100;
const MAX_EPISODE_MOVES = 200;

const TILE_CODES: Record<string, Tile> = { '.': 'free', '#': 'wall', C: 'cliff', S: 'start', G: 'goal' };

export function parseWorld(map: string[]): GridWorld {
	const tiles = map.join('').split('').map((code) => TILE_CODES[code]);
	return {
		cols: map[0].length,
		rows: map.length,
		tiles,
		start: tiles.indexOf('start'),
		goal: tiles.indexOf('goal'),
	};
}

export const WORLDS = {
	cliff: parseWorld(['............', '............', '............', 'SCCCCCCCCCCG']),
	rooms: parseWorld(['S...#....', '....#....', '.........', '....#....', '##.##....', '....#....', '....#...G']),
};

export function isTerminal(world: GridWorld, cell: number): boolean {
	return world.tiles[cell] === 'goal' || world.tiles[cell] === 'cliff';
}

/** Moves one cell. Edges and walls leave the agent where it is. */
export function transition(world: GridWorld, cell: number, action: number): { next: number; reward: number } {
	const [dx, dy] = ACTIONS[action];
	const col = (cell % world.cols) + dx;
	const row = Math.floor(cell / world.cols) + dy;
	const inside = col >= 0 && col < world.cols && row >= 0 && row < world.rows;
	const target = row * world.cols + col;
	const next = inside && world.tiles[target] !== 'wall' ? target : cell;
	return { next, reward: world.tiles[next] === 'cliff' ? CLIFF_REWARD : STEP_REWARD };
}

export function greedyAction(values: number[]): number {
	return values.indexOf(Math.max(...values));
}

/** Explores with probability epsilon, and breaks value ties at random. */
function chooseAction(values: number[], epsilon: number, rng: () => number): number {
	if (rng() < epsilon) return Math.floor(rng() * values.length);
	const best = Math.max(...values);
	const ties = values.flatMap((v, a) => (v === best ? [a] : []));
	return ties[Math.floor(rng() * ties.length)];
}

export function initRl(world: GridWorld, seed = 13): RlState {
	return {
		q: world.tiles.map(() => [0, 0, 0, 0]),
		position: world.start,
		action: null,
		episode: 0,
		moves: 0,
		episodeReturn: 0,
		returns: [],
		lastUpdate: null,
		rng: createRng(seed),
	};
}

/** One agent move and one temporal-difference update. */
export function rlStep(state: RlState, world: GridWorld, params: RlParams): RlState {
	const { alpha, gamma, epsilon, algorithm } = params;
	const cell = state.position;
	const action = state.action ?? chooseAction(state.q[cell], epsilon, state.rng);
	const { next, reward } = transition(world, cell, action);
	const terminal = isTerminal(world, next);
	const nextAction = terminal ? null : chooseAction(state.q[next], epsilon, state.rng);

	// The only difference between the two methods: which next value they bootstrap from.
	let bootstrap = 0;
	if (nextAction !== null) {
		bootstrap = algorithm === 'sarsa' ? state.q[next][nextAction] : Math.max(...state.q[next]);
	}
	const target = reward + gamma * bootstrap;

	const q = state.q.slice();
	q[cell] = state.q[cell].slice();
	q[cell][action] += alpha * (target - q[cell][action]);

	const moves = state.moves + 1;
	const episodeReturn = state.episodeReturn + reward;
	const base = { ...state, q, lastUpdate: { reward, target } };
	if (terminal || moves >= MAX_EPISODE_MOVES) {
		return {
			...base,
			position: world.start,
			action: null,
			episode: state.episode + 1,
			moves: 0,
			episodeReturn: 0,
			returns: [...state.returns, episodeReturn],
		};
	}
	return { ...base, position: next, action: algorithm === 'sarsa' ? nextAction : null, moves, episodeReturn };
}

/** Follows the greedy policy from the start, stopping at a terminal cell or a loop. */
export function greedyPath(world: GridWorld, q: number[][]): number[] {
	const path = [world.start];
	while (!isTerminal(world, path.at(-1)!)) {
		const { next } = transition(world, path.at(-1)!, greedyAction(q[path.at(-1)!]));
		if (path.includes(next)) break;
		path.push(next);
	}
	return path;
}
