import type { Dataset } from '../../types/ml';
import { kMeansStep, nearestCentroid, type Centroid, type KMeansState } from '../ml/kmeans';
import { createRng } from '../ml/random';
import type { PlotColors } from './canvas';

const K = 4;
const MAX_STEPS = 20;
const FADE_S = 1.2;
const STEP_S = 1.8;
const HOLD_S = 4.5;
/** The last part of a step, when points take the color of their new cluster. */
const RECOLOR_FROM = 0.7;

/** A finished k-means run, kept step by step so it can be replayed slowly. */
export interface HeroRun {
	data: Dataset;
	steps: KMeansState[];
}

export interface HeroPose {
	centroids: Centroid[];
	/** Index of the step whose assignments are showing. */
	step: number;
	/** How far points have moved to the next step's colors, from 0 to 1. */
	recolor: number;
	opacity: number;
}

const easeInOut = (p: number) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** Clusters sit towards the corners, so the centroids have room to travel. */
function heroDataset(n: number, rng: () => number): Dataset {
	const gaussian = () => (rng() + rng() + rng() - 1.5) * 0.8;
	const corners = [
		{ x: -0.68, y: -0.5 },
		{ x: 0.68, y: -0.5 },
		{ x: -0.68, y: 0.5 },
		{ x: 0.68, y: 0.5 },
	];
	return Array.from({ length: n }, (_, i) => {
		const corner = corners[i % K];
		return { x: corner.x + gaussian() * 0.3, y: corner.y + gaussian() * 0.42, label: 0 };
	});
}

/** Runs real k-means from centroids bunched in the middle, so they spread out to find the clusters. */
export function createHeroRun(seed: number, n: number): HeroRun {
	const rng = createRng(seed);
	const data = heroDataset(n, rng);
	const centroids = Array.from({ length: K }, () => ({ x: (rng() - 0.5) * 0.5, y: (rng() - 0.5) * 0.5 }));
	const first: KMeansState = {
		centroids,
		assignments: data.map((p) => nearestCentroid(centroids, p.x, p.y)),
		iteration: 0,
		inertia: 0,
		inertiaHistory: [],
		converged: false,
	};
	const steps = [first];
	while (!steps[steps.length - 1].converged && steps.length <= MAX_STEPS) {
		steps.push(kMeansStep(steps[steps.length - 1], data));
	}
	return { data, steps };
}

export const heroDuration = (run: HeroRun) => FADE_S * 2 + (run.steps.length - 1) * STEP_S + HOLD_S;

/** Where everything is `seconds` into a run. */
export function heroPose(run: HeroRun, seconds: number): HeroPose {
	const last = run.steps.length - 1;
	const duration = heroDuration(run);
	const opacity = clamp01(Math.min(seconds / FADE_S, (duration - seconds) / FADE_S));
	const progress = clamp01((seconds - FADE_S) / (last * STEP_S)) * last;
	const step = Math.min(Math.floor(progress), Math.max(last - 1, 0));
	const along = last === 0 ? 0 : progress - step;

	const from = run.steps[step];
	const to = run.steps[Math.min(step + 1, last)];
	const glide = easeInOut(along);
	return {
		centroids: from.centroids.map((c, i) => ({
			x: c.x + (to.centroids[i].x - c.x) * glide,
			y: c.y + (to.centroids[i].y - c.y) * glide,
		})),
		step,
		recolor: clamp01((along - RECOLOR_FROM) / (1 - RECOLOR_FROM)),
		opacity,
	};
}

/** The finished result, for visitors who don't want motion. */
export const heroFinalPose = (run: HeroRun): HeroPose => ({
	...heroPose(run, heroDuration(run) - FADE_S),
	opacity: 1,
});

export function drawHero(ctx: CanvasRenderingContext2D, w: number, h: number, run: HeroRun, pose: HeroPose, colors: PlotColors) {
	ctx.clearRect(0, 0, w, h);
	const px = (x: number) => ((x + 1) / 2) * w;
	const py = (y: number) => ((y + 1) / 2) * h;
	const from = run.steps[pose.step].assignments;
	const to = run.steps[Math.min(pose.step + 1, run.steps.length - 1)].assignments;
	const r = Math.max(2.5, Math.min(w, h) * 0.013);

	run.data.forEach((p, i) => {
		const owner = pose.recolor > 0.5 ? to[i] : from[i];
		const x = px(p.x);
		const y = py(p.y);
		const centroid = pose.centroids[owner];

		ctx.globalAlpha = pose.opacity * 0.22;
		ctx.strokeStyle = colors.series[owner];
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(x, y);
		ctx.lineTo(px(centroid.x), py(centroid.y));
		ctx.stroke();

		const changing = from[i] !== to[i];
		const shares = changing ? [[from[i], 1 - pose.recolor], [to[i], pose.recolor]] : [[from[i], 1]];
		for (const [cluster, share] of shares) {
			ctx.globalAlpha = pose.opacity * share;
			ctx.fillStyle = colors.series[cluster];
			ctx.beginPath();
			ctx.arc(x, y, r, 0, Math.PI * 2);
			ctx.fill();
		}
	});

	pose.centroids.forEach((c, i) => {
		ctx.globalAlpha = pose.opacity;
		ctx.strokeStyle = colors.series[i];
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.arc(px(c.x), py(c.y), r * 3.5, 0, Math.PI * 2);
		ctx.stroke();
		ctx.fillStyle = colors.ink;
		ctx.beginPath();
		ctx.arc(px(c.x), py(c.y), r * 1.1, 0, Math.PI * 2);
		ctx.fill();
	});
	ctx.globalAlpha = 1;
}
