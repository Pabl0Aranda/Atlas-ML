import type { Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec } from '../../../types/viz';
import {
	currentEpsilon,
	dqnStep,
	initDqn,
	qTable,
	qValues,
	REWARD_SCALE,
	type DqnParams,
	type DqnState,
} from '../../ml/dqn';
import { greedyPath } from '../../ml/tabularRl';
import { fixed } from '../scene';
import { ACTION_NAMES, AVERAGE_WINDOW, cellAt, cells, movingAverage, worldOf } from './rl';

const MAX_MOVES = 12000;
const EPSILON_DECAY = 4000;

const toParams = (p: Hyperparameters): DqnParams => ({
	learningRate: p.learningRate,
	gamma: p.gamma,
	epsilonEnd: p.epsilonEnd,
	epsilonDecay: EPSILON_DECAY,
	targetSync: p.targetSync,
});

export const deepQNetwork: AlgorithmSpec<DqnState> = {
	datasets: [],
	editable: false,
	params: [
		{ key: 'world', label: 'World', value: 0, options: [{ label: 'Cliff', value: 0 }, { label: 'Rooms', value: 1 }] },
		{ key: 'learningRate', label: 'Learning rate α', value: 1, min: 0.1, max: 2, step: 0.1 },
		{ key: 'gamma', label: 'Discount γ', value: 0.98, min: 0.8, max: 0.99, step: 0.01 },
		{ key: 'epsilonEnd', label: 'Final exploration ε', value: 0.1, min: 0.01, max: 0.5, step: 0.01 },
		{ key: 'targetSync', label: 'Target sync (moves)', value: 200, min: 20, max: 1000, step: 20 },
	],
	labeled: false,
	speed: 30,
	thumbnailSteps: MAX_MOVES,
	legend: (p) => (p.world === 1 ? ['Learned value'] : ['Learned value', 'Cliff']),
	init: (_, p) => initDqn(worldOf(p)),
	step: (s, _, p) => dqnStep(s, worldOf(p), toParams(p)),
	isDone: (s) => s.totalMoves >= MAX_MOVES || !Number.isFinite(s.loss),
	scene(s, _, p) {
		const world = worldOf(p);
		const q = qTable(s.online, world);
		return {
			points: [],
			grid: {
				cols: world.cols,
				rows: world.rows,
				cells: cells(q, world),
				agent: s.position,
				path: s.episode > 0 ? greedyPath(world, q) : [],
			},
		};
	},
	metrics(s, _, p) {
		const world = worldOf(p);
		const path = greedyPath(world, qTable(s.online, world));
		return [
			{ label: 'Episode', value: String(s.episode) },
			{ label: 'Exploration ε', value: currentEpsilon(s, toParams(p)).toFixed(2) },
			{ label: 'Replay memories', value: String(s.buffer.length) },
			{ label: 'Greedy route', value: path.at(-1) === world.goal ? `${path.length - 1} moves` : 'not found' },
		];
	},
	caption(s) {
		if (!Number.isFinite(s.loss)) return 'The updates diverged. Lower the learning rate and reset.';
		if (s.totalMoves === 0) return 'Random network weights, so these values mean nothing yet. Press Play: training takes about a minute.';
		if (s.buffer.length < 200) return 'Collecting random experience in the replay buffer before training begins.';
		return `Move ${s.totalMoves}. Each move trains on 32 random memories. TD loss ${fixed(s.loss, 5)}.`;
	},
	series: (s) => ({ label: `Return per episode (average of ${AVERAGE_WINDOW})`, values: movingAverage(s.returns) }),
	describe(s, probe, p) {
		const world = worldOf(p);
		const values = qValues(s.online, world, cellAt(world, probe));
		return values.map((v, a) => `${ACTION_NAMES[a]} ${(v / REWARD_SCALE).toFixed(1)}`).join(' · ');
	},
};
