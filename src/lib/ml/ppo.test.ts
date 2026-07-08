import { describe, expect, it } from 'vitest';
import { actionProbabilities, computeAdvantages, initPpo, ppoStep, softmax, type PpoParams } from './ppo';
import { greedyPath, WORLDS } from './tabularRl';

const params: PpoParams = { learningRate: 0.3, gamma: 0.98, lambda: 0.95, clip: 0.2, entropyBonus: 0.03, rollout: 512, epochs: 4 };
const world = WORLDS.rooms;

const run = (moves: number) => {
	let s = initPpo(world);
	for (let i = 0; i < moves; i++) s = ppoStep(s, world, params);
	return s;
};

describe('PPO mechanics', () => {
	it('computes a stable softmax', () => {
		const p = softmax([1000, 1000]);
		expect(p).toEqual([0.5, 0.5]);
	});

	it('starts near a uniform policy', () => {
		const s = initPpo(world);
		for (const p of actionProbabilities(s.policy, world, world.start)) expect(p).toBeCloseTo(0.25, 1);
	});

	it('computes GAE by hand', () => {
		const samples = [
			{ reward: 1, value: 0, terminal: false, ended: false },
			{ reward: 1, value: 0, terminal: true, ended: true },
		];
		const [first, last] = computeAdvantages(samples, [0, 5], 0.5, 0.5);
		expect(last).toBe(1);
		expect(first).toBe(1 + 0.25 * 1);
	});

	it('resets the running advantage between episodes', () => {
		const samples = [
			{ reward: 0, value: 0, terminal: true, ended: true },
			{ reward: 9, value: 0, terminal: true, ended: true },
		];
		expect(computeAdvantages(samples, [0, 0], 0.9, 0.9)[0]).toBe(0);
	});

	it('bootstraps through a time limit but not a terminal cell', () => {
		const timeLimit = [{ reward: 0, value: 0, terminal: false, ended: true }];
		expect(computeAdvantages(timeLimit, [2], 0.5, 0.5)[0]).toBe(1);
	});

	it('updates once per rollout and then clears it', () => {
		const s = run(512);
		expect(s.updates).toBe(1);
		expect(s.rollout).toHaveLength(0);
		expect(run(100).rollout).toHaveLength(100);
	});
});

describe('PPO learning', () => {
	it('solves the rooms world', { timeout: 30000 }, () => {
		const s = run(80000);
		const path = greedyPath(world, world.tiles.map((_, c) => actionProbabilities(s.policy, world, c)));
		expect(path.at(-1)).toBe(world.goal);
		expect(path.length - 1).toBe(14);
		expect(s.returns.at(-1)).toBeGreaterThan(-40);
	});
});
