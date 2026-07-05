import { describe, expect, it } from 'vitest';
import { currentEpsilon, dqnStep, encode, initDqn, qTable, type DqnParams } from './dqn';
import { greedyPath, WORLDS } from './tabularRl';

const params: DqnParams = { learningRate: 1, gamma: 0.98, epsilonEnd: 0.1, epsilonDecay: 4000, targetSync: 200 };
const world = WORLDS.cliff;

const run = (moves: number, p = params) => {
	let s = initDqn(world);
	for (let i = 0; i < moves; i++) s = dqnStep(s, world, p);
	return s;
};

describe('DQN mechanics', () => {
	it('encodes a cell as one column and one row', () => {
		const features = encode(world, world.cols + 2);
		expect(features).toHaveLength(world.cols + world.rows);
		expect(features.filter(Boolean)).toHaveLength(2);
		expect(features[2]).toBe(1);
		expect(features[world.cols + 1]).toBe(1);
	});

	it('decays exploration to its floor', () => {
		expect(currentEpsilon(initDqn(world), params)).toBe(1);
		expect(currentEpsilon({ ...initDqn(world), totalMoves: 2000 }, params)).toBeCloseTo(0.55);
		expect(currentEpsilon({ ...initDqn(world), totalMoves: 9000 }, params)).toBeCloseTo(0.1);
	});

	it('freezes the target network between syncs', () => {
		const before = run(250);
		const after = dqnStep(before, world, params);
		expect(after.target).toBe(before.target);
		const synced = run(399);
		const atSync = dqnStep(synced, world, params);
		expect(atSync.target).not.toBe(synced.target);
		expect(atSync.target).toEqual(atSync.online);
	});

	it('stores every move in the replay buffer', () => {
		expect(run(300).buffer).toHaveLength(300);
	});
});

describe('DQN learning', () => {
	it('finds the optimal cliff route', { timeout: 60000 }, () => {
		const s = run(12000);
		const path = greedyPath(world, qTable(s.online, world));
		expect(path.at(-1)).toBe(world.goal);
		expect(path.length - 1).toBe(13);
	});
});
