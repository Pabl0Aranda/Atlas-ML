import { describe, expect, it } from 'vitest';
import { kMeansStep } from '../ml/kmeans';
import { createHeroRun, heroDuration, heroFinalPose, heroPose } from './heroMotion';

describe('hero k-means run', () => {
	for (const seed of [3, 4, 5, 6, 7, 8]) {
		it(`converges with real k-means for seed ${seed}`, () => {
			const run = createHeroRun(seed, 96);
			const last = run.steps[run.steps.length - 1];
			expect(last.converged).toBe(true);
			expect(run.steps.length).toBeGreaterThan(1);
			expect(kMeansStep(last, run.data).assignments).toEqual(last.assignments);
		});
	}

	it('repeats for a seed and changes between seeds', () => {
		expect(createHeroRun(3, 48)).toEqual(createHeroRun(3, 48));
		expect(createHeroRun(3, 48).data).not.toEqual(createHeroRun(4, 48).data);
	});

	it('uses the requested number of points, for lighter phones', () => {
		expect(createHeroRun(3, 48).data).toHaveLength(48);
	});
});

describe('heroPose', () => {
	const run = createHeroRun(3, 96);
	const duration = heroDuration(run);
	const final = run.steps[run.steps.length - 1];

	it('fades in and out, and is fully visible between', () => {
		expect(heroPose(run, 0).opacity).toBe(0);
		expect(heroPose(run, duration).opacity).toBeCloseTo(0);
		expect(heroPose(run, duration / 2).opacity).toBe(1);
	});

	it('starts on the first centroids and ends on the converged ones', () => {
		expect(heroPose(run, 0).centroids).toEqual(run.steps[0].centroids);
		expect(heroPose(run, duration - 2).centroids).toEqual(final.centroids);
	});

	it('stays finite and in range through the whole run', () => {
		for (let t = -1; t <= duration + 1; t += 0.13) {
			const pose = heroPose(run, t);
			expect(pose.step).toBeGreaterThanOrEqual(0);
			expect(pose.step).toBeLessThan(run.steps.length);
			expect(pose.recolor).toBeGreaterThanOrEqual(0);
			expect(pose.recolor).toBeLessThanOrEqual(1);
			expect(pose.opacity).toBeGreaterThanOrEqual(0);
			expect(pose.opacity).toBeLessThanOrEqual(1);
			for (const c of pose.centroids) expect(Number.isFinite(c.x + c.y)).toBe(true);
		}
	});

	it('shows the finished result to visitors who avoid motion', () => {
		const pose = heroFinalPose(run);
		expect(pose.opacity).toBe(1);
		expect(pose.centroids).toEqual(final.centroids);
	});
});
