import { describe, expect, it } from 'vitest';
import { descentStep, gradientSize, lowestLoss, startDescent, SURFACES, type DescentState } from './gradientDescent';

const run = (surface: number, alpha: number, beta: number, x: number, y: number, steps: number): DescentState => {
	let state = startDescent(SURFACES[surface], x, y);
	for (let i = 0; i < steps; i++) state = descentStep(state, SURFACES[surface], alpha, beta);
	return state;
};

describe('gradient descent', () => {
	it('has gradients that match the slope of each loss', () => {
		const h = 1e-5;
		for (const surface of SURFACES) {
			for (const [x, y] of [[0.3, -0.4], [-0.7, 0.2], [0.1, 0.9]]) {
				const [gx, gy] = surface.gradient(x, y);
				expect(gx).toBeCloseTo((surface.loss(x + h, y) - surface.loss(x - h, y)) / (2 * h), 4);
				expect(gy).toBeCloseTo((surface.loss(x, y + h) - surface.loss(x, y - h)) / (2 * h), 4);
			}
		}
	});

	it('reaches the bottom of the bowl with a moderate learning rate', () => {
		const state = run(0, 0.1, 0, -0.8, 0.6, 100);
		expect(Math.hypot(state.x, state.y)).toBeLessThan(1e-3);
		expect(state.losses.every((loss, i) => i === 0 || loss < state.losses[i - 1])).toBe(true);
	});

	it('diverges when the step overshoots by more than it gains', () => {
		const state = run(1, 0.25, 0, -0.8, 0.6, 50);
		expect(state.diverged).toBe(true);
		expect(state.path.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))).toBe(true);
	});

	it('stops changing once diverged', () => {
		const state = run(1, 0.25, 0, -0.8, 0.6, 50);
		expect(descentStep(state, SURFACES[1], 0.25, 0)).toBe(state);
	});

	it('settles in the valley nearest the start on the two-valley surface', () => {
		expect(run(2, 0.05, 0, 0.9, 0.5, 300).x).toBeGreaterThan(0.5);
		expect(run(2, 0.05, 0, -0.3, 0.5, 300).x).toBeLessThan(-0.5);
	});

	it('gets further along the narrow valley with momentum in the same number of steps', () => {
		const plain = run(1, 0.02, 0, -0.9, 0.6, 60);
		const momentum = run(1, 0.02, 0.8, -0.9, 0.6, 60);
		expect(momentum.losses.at(-1)!).toBeLessThan(plain.losses.at(-1)!);
	});

	it('finds the floor of each surface at or below every visited loss', () => {
		for (const surface of SURFACES) expect(lowestLoss(surface)).toBeLessThanOrEqual(surface.loss(0.5, 0.25));
		expect(gradientSize(SURFACES[0], 0, 0)).toBe(0);
	});
});
