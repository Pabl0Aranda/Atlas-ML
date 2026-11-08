import { describe, expect, it } from 'vitest';
import type { Hyperparameters } from '../../types/ml';
import { PRESETS } from './presets';
import { defaultParams } from './scene';
import { gradientDescent as spec } from './specs/gradientDescent';

const presetParams = (label: string, change: Hyperparameters = {}): Hyperparameters => {
	const preset = PRESETS['gradient-descent']!.find((p) => p.label === label)!;
	return { ...defaultParams(spec), ...preset.params, ...change };
};

const finish = (params: Hyperparameters) => {
	let state = spec.init([], params);
	while (!spec.isDone(state, params)) state = spec.step(state, [], params);
	return state;
};

// The gradient descent page and its preset notes quote these figures.
describe('gradient descent presets', () => {
	it('zigzag settles after 39 steps, and a slightly larger rate diverges after 29', () => {
		expect(finish(presetParams('Zigzag'))).toMatchObject({ step: 39, diverged: false });
		expect(finish(presetParams('Too large'))).toMatchObject({ step: 29, diverged: true });
	});

	it('momentum settles in 54 steps, where plain descent is still going after 200', () => {
		expect(finish(presetParams('Momentum')).step).toBe(54);
		const plain = finish(presetParams('Momentum', { beta: 0 }));
		expect(plain.step).toBe(200);
		expect(spec.caption(plain, [], presetParams('Momentum', { beta: 0 }))).toMatch(/^Stopped/);
	});

	it('ends in the valley nearest the start, not the deepest one', () => {
		const right = finish(presetParams('Local minimum'));
		const left = finish(presetParams('Local minimum', { startX: -0.8 }));
		expect(right.x).toBeGreaterThan(0);
		expect(right.losses.at(-1)).toBeCloseTo(0.25, 2);
		expect(left.x).toBeLessThan(0);
		expect(left.losses.at(-1)).toBeCloseTo(0.04, 2);
	});

	it('crawls along the curved valley without momentum and settles in 78 steps with it', () => {
		expect(finish(presetParams('Curved valley')).step).toBe(200);
		expect(finish(presetParams('Curved valley', { beta: 0.8 })).step).toBe(78);
	});

	it('names the zigzag as it happens', () => {
		const params = presetParams('Zigzag');
		let state = spec.init([], params);
		for (let i = 0; i < 3; i++) state = spec.step(state, [], params);
		expect(spec.caption(state, [], params)).toMatch(/^Zigzag/);
	});
});
