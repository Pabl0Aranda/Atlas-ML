import type { Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec } from '../../../types/viz';
import { actionProbabilities, initPpo, ppoStep, stateValue, type PpoParams, type PpoState } from '../../ml/ppo';
import { greedyPath, isTerminal, type GridWorld } from '../../ml/tabularRl';
import { percent } from '../scene';
import { ACTION_NAMES, AVERAGE_WINDOW, cellAt, cells, movingAverage, worldOf } from './rl';

const MAX_MOVES = 80000;

const toParams = (p: Hyperparameters): PpoParams => ({
	learningRate: p.learningRate,
	gamma: 0.98,
	lambda: 0.95,
	clip: p.clip,
	entropyBonus: p.entropyBonus,
	rollout: p.rollout,
	epochs: 4,
});

const policyTable = (s: PpoState, world: GridWorld) => world.tiles.map((_, c) => actionProbabilities(s.policy, world, c));

/** Arrows come from the policy, and shading from the critic. */
function criticCells(s: PpoState, world: GridWorld) {
	const values = world.tiles.map((_, c) => stateValue(s.critic, world, c));
	const open = values.filter((_, c) => !isTerminal(world, c) && world.tiles[c] !== 'wall');
	const low = Math.min(...open);
	const range = Math.max(...open) - low || 1;
	return cells(policyTable(s, world), world).map((cell, c) => ({
		...cell,
		value: cell.arrow === undefined ? 0 : (values[c] - low) / range,
	}));
}

export const proximalPolicyOptimization: AlgorithmSpec<PpoState> = {
	datasets: [],
	editable: false,
	// Each update trains both networks in place, for speed.
	snapshot: (s) => ({ ...s, policy: structuredClone(s.policy), critic: structuredClone(s.critic) }),
	params: [
		{ key: 'world', label: 'World', value: 1, options: [{ label: 'Rooms', value: 1 }, { label: 'Cliff', value: 0 }] },
		{ key: 'learningRate', label: 'Learning rate α', value: 0.3, min: 0.1, max: 2, step: 0.1 },
		{ key: 'clip', label: 'Clip range ε', value: 0.2, min: 0.05, max: 0.5, step: 0.05 },
		{ key: 'entropyBonus', label: 'Entropy bonus', value: 0.03, min: 0, max: 0.1, step: 0.01 },
		{ key: 'rollout', label: 'Rollout (moves)', value: 512, min: 256, max: 2048, step: 256 },
	],
	labeled: false,
	speed: 400,
	thumbnailSteps: MAX_MOVES,
	legend: (p) => (p.world === 1 ? ['Critic value'] : ['Critic value', 'Cliff']),
	init: (_, p) => initPpo(worldOf(p)),
	step: (s, _, p) => ppoStep(s, worldOf(p), toParams(p)),
	isDone: (s) => s.totalMoves >= MAX_MOVES,
	scene(s, _, p) {
		const world = worldOf(p);
		return {
			points: [],
			grid: {
				cols: world.cols,
				rows: world.rows,
				cells: criticCells(s, world),
				agent: s.position,
				path: s.updates > 0 ? greedyPath(world, policyTable(s, world)) : [],
			},
		};
	},
	metrics(s, _, p) {
		const world = worldOf(p);
		const path = greedyPath(world, policyTable(s, world));
		return [
			{ label: 'Policy updates', value: String(s.updates) },
			{ label: 'Policy entropy', value: s.entropy.toFixed(2) },
			{ label: 'Clipped samples', value: s.updates ? percent(s.clippedShare) : '–' },
			{ label: 'Likeliest route', value: path.at(-1) === world.goal ? `${path.length - 1} moves` : 'not found' },
		];
	},
	caption(s, _, p) {
		if (s.updates === 0) return `The policy is nearly uniform. It collects ${p.rollout} moves before its first update.`;
		if (s.entropy < 0.05 && s.returns.at(-1)! <= -100) return 'The policy has become certain of a bad habit. With no exploration left, it cannot recover.';
		return `Move ${s.totalMoves}. Entropy falls as the policy grows more certain.`;
	},
	series: (s) => ({ label: `Return per episode (average of ${AVERAGE_WINDOW})`, values: movingAverage(s.returns) }),
	describe(s, probe, p) {
		const world = worldOf(p);
		return actionProbabilities(s.policy, world, cellAt(world, probe))
			.map((v, a) => `${ACTION_NAMES[a]} ${Math.round(v * 100)}%`)
			.join(' · ');
	},
};
