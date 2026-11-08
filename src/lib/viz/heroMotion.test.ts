import { describe, expect, it } from 'vitest';
import { kMeansStep } from '../ml/kmeans';
import { addHeroPoint, createHeroRun, HERO_RESUME_AT, heroDuration, heroFinalPose, heroPose, MAX_ADDED, toPlot } from './heroMotion';

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

describe('adding points to the hero run', () => {
	const run = createHeroRun(3, 48);
	const final = heroFinalPose(run);

	it('adds the point and runs k-means again from the centroids on screen', () => {
		const next = addHeroPoint(run, final.centroids, { x: 0.9, y: -0.9 });
		expect(next.data).toHaveLength(49);
		expect(next.data.at(-1)).toMatchObject({ x: 0.9, y: -0.9 });
		expect(next.steps[0].centroids).toEqual(final.centroids);
		const last = next.steps.at(-1)!;
		expect(last.converged).toBe(true);
		expect(kMeansStep(last, next.data).assignments).toEqual(last.assignments);
	});

	it('marks which points the visitor added', () => {
		expect(run.added).toBe(0);
		const once = addHeroPoint(run, final.centroids, { x: 0, y: 0 });
		const twice = addHeroPoint(once, heroFinalPose(once).centroids, { x: 0.1, y: 0.1 });
		expect(once.added).toBe(1);
		expect(twice.added).toBe(2);
	});

	it('pulls the nearest centroid toward a far-away point', () => {
		const corner = { x: 0.98, y: 0.98 };
		const dist = (c: { x: number; y: number }) => Math.hypot(c.x - corner.x, c.y - corner.y);
		let next = run;
		for (let i = 0; i < 8; i++) next = addHeroPoint(next, heroFinalPose(next).centroids, corner);
		const before = Math.min(...final.centroids.map(dist));
		const after = Math.min(...heroFinalPose(next).centroids.map(dist));
		expect(after).toBeLessThan(before);
	});

	it('keeps points inside the plot and caps how many a visitor can add', () => {
		const outside = addHeroPoint(run, final.centroids, { x: 3, y: -3 });
		expect(outside.data.at(-1)).toMatchObject({ x: 1, y: -1 });
		let next = run;
		for (let i = 0; i < MAX_ADDED + 5; i++) next = addHeroPoint(next, heroFinalPose(next).centroids, { x: (i % 10) / 10, y: 0.5 });
		expect(next.added).toBe(MAX_ADDED);
		expect(next.data).toHaveLength(48 + MAX_ADDED);
		expect(next.data.slice(0, 48)).toEqual(run.data);
	});

	it('starts a run after a click fully visible, without fading in again', () => {
		const next = addHeroPoint(run, final.centroids, { x: 0.5, y: 0.5 });
		expect(heroPose(next, HERO_RESUME_AT).opacity).toBe(1);
		expect(heroPose(next, HERO_RESUME_AT).centroids).toEqual(final.centroids);
	});

	it('converts canvas pixels to plot coordinates', () => {
		expect(toPlot(0, 0, 400, 300)).toEqual({ x: -1, y: -1 });
		expect(toPlot(400, 300, 400, 300)).toEqual({ x: 1, y: 1 });
		expect(toPlot(200, 150, 400, 300)).toEqual({ x: 0, y: 0 });
	});
});
