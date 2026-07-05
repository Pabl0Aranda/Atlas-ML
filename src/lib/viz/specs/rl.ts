import type { Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec, GridCell } from '../../../types/viz';
import {
	greedyAction,
	greedyPath,
	initRl,
	isTerminal,
	rlStep,
	WORLDS,
	type GridWorld,
	type RlAlgorithm,
	type RlState,
} from '../../ml/tabularRl';
import { fixed } from '../scene';

const MAX_EPISODES = 500;
export const AVERAGE_WINDOW = 10;
export const ACTION_NAMES = ['up', 'right', 'down', 'left'];

export const worldOf = (p: Hyperparameters): GridWorld => (p.world === 1 ? WORLDS.rooms : WORLDS.cliff);

/** The grid cell under a location in the plot domain. */
export function cellAt(world: GridWorld, probe: { x: number; y: number }): number {
	const col = Math.min(world.cols - 1, Math.floor(((probe.x + 1) / 2) * world.cols));
	const row = Math.min(world.rows - 1, Math.floor(((1 - probe.y) / 2) * world.rows));
	return row * world.cols + col;
}

export function cells(q: number[][], world: GridWorld): GridCell[] {
	const values = q.map((row) => Math.max(...row));
	const open = values.filter((_, i) => !isTerminal(world, i) && world.tiles[i] !== 'wall');
	const low = Math.min(...open);
	const range = Math.max(...open) - low || 1;
	return world.tiles.map((tile, i) => {
		const visited = q[i].some((v) => v !== 0);
		const walkable = tile === 'free' || tile === 'start';
		return {
			tile,
			value: walkable && visited ? (values[i] - low) / range : 0,
			arrow: walkable && visited ? greedyAction(q[i]) : undefined,
		};
	});
}

export function movingAverage(values: number[]): number[] {
	return values.map((_, i) => {
		const window = values.slice(Math.max(0, i - AVERAGE_WINDOW + 1), i + 1);
		return window.reduce((a, b) => a + b, 0) / window.length;
	});
}

function createSpec(algorithm: RlAlgorithm): AlgorithmSpec<RlState> {
	const toParams = (p: Hyperparameters) => ({ algorithm, alpha: p.alpha, gamma: p.gamma, epsilon: p.epsilon });
	return {
		datasets: [],
		editable: false,
		params: [
			{ key: 'world', label: 'World', value: 0, options: [{ label: 'Cliff', value: 0 }, { label: 'Rooms', value: 1 }] },
			{ key: 'alpha', label: 'Learning rate α', value: 0.5, min: 0.05, max: 1, step: 0.05 },
			{ key: 'gamma', label: 'Discount γ', value: 1, min: 0.5, max: 1, step: 0.01 },
			{ key: 'epsilon', label: 'Exploration ε', value: 0.1, min: 0, max: 0.5, step: 0.01 },
		],
		labeled: false,
		speed: 60,
		thumbnailSteps: 12000,
		legend: (p) => (p.world === 1 ? ['Learned value'] : ['Learned value', 'Cliff']),
		init: (_, p) => initRl(worldOf(p)),
		step: (s, _, p) => rlStep(s, worldOf(p), toParams(p)),
		isDone: (s) => s.episode >= MAX_EPISODES,
		scene(s, _, p) {
			const world = worldOf(p);
			return {
				points: [],
				grid: {
					cols: world.cols,
					rows: world.rows,
					cells: cells(s.q, world),
					agent: s.position,
					path: s.episode > 0 ? greedyPath(world, s.q) : [],
				},
			};
		},
		metrics(s, _, p) {
			const world = worldOf(p);
			const path = greedyPath(world, s.q);
			const reachesGoal = path.at(-1) === world.goal;
			return [
				{ label: 'Episode', value: String(s.episode) },
				{ label: 'Moves this episode', value: String(s.moves) },
				{ label: 'Last return', value: s.returns.length ? String(s.returns.at(-1)) : '–' },
				{ label: 'Greedy route', value: reachesGoal ? `${path.length - 1} moves` : 'not found' },
			];
		},
		caption(s) {
			if (!s.lastUpdate) return 'All values start at zero. Step to make one move and one update, or press Play.';
			const { reward, target } = s.lastUpdate;
			return `Last move: reward ${reward}, TD target ${fixed(target, 2)}. The line traces the current greedy route.`;
		},
		series: (s) => ({ label: `Return per episode (average of ${AVERAGE_WINDOW})`, values: movingAverage(s.returns) }),
		describe(s, probe, p) {
			const world = worldOf(p);
			const values = s.q[cellAt(world, probe)];
			return values.map((v, a) => `${ACTION_NAMES[a]} ${v.toFixed(1)}`).join(' · ');
		},
	};
}

export const qLearning = createSpec('q-learning');
export const sarsa = createSpec('sarsa');
