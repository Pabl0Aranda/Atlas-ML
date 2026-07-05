import { describe, expect, it } from 'vitest';
import { greedyPath, initRl, rlStep, transition, WORLDS, type RlAlgorithm, type RlState } from './tabularRl';

const train = (algorithm: RlAlgorithm, world = WORLDS.cliff, episodes = 500): RlState => {
	let s = initRl(world);
	const params = { algorithm, alpha: 0.5, gamma: 1, epsilon: 0.1 };
	while (s.episode < episodes) s = rlStep(s, world, params);
	return s;
};

const average = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

describe('gridworld', () => {
	it('blocks edges and walls and penalizes the cliff', () => {
		const { cliff, rooms } = WORLDS;
		expect(transition(cliff, cliff.start, 3)).toEqual({ next: cliff.start, reward: -1 });
		expect(transition(cliff, cliff.start, 1).reward).toBe(-100);
		expect(transition(rooms, 3, 1).next).toBe(3);
	});
});

describe('tabular RL', () => {
	it('applies the TD update exactly', () => {
		const world = WORLDS.cliff;
		const s = rlStep(initRl(world), world, { algorithm: 'q-learning', alpha: 0.5, gamma: 1, epsilon: 0 });
		expect(s.q.flat().filter((v) => v !== 0).length).toBe(1);
		expect(Math.min(...s.q.flat())).toBeLessThan(0);
	});

	it('Q-learning finds the optimal path along the cliff edge', () => {
		const s = train('q-learning');
		const path = greedyPath(WORLDS.cliff, s.q);
		expect(path.length - 1).toBe(13);
		expect(path.at(-1)).toBe(WORLDS.cliff.goal);
	});

	it('SARSA takes a longer, safer path and earns more while exploring', () => {
		const sarsa = train('sarsa');
		const path = greedyPath(WORLDS.cliff, sarsa.q);
		expect(path.at(-1)).toBe(WORLDS.cliff.goal);
		expect(path.length - 1).toBeGreaterThan(13);
		const late = (s: RlState) => average(s.returns.slice(-200));
		expect(late(sarsa)).toBeGreaterThan(late(train('q-learning')));
	});

	it('solves the rooms world through its doorways', () => {
		const path = greedyPath(WORLDS.rooms, train('q-learning', WORLDS.rooms).q);
		expect(path.at(-1)).toBe(WORLDS.rooms.goal);
	});
});
