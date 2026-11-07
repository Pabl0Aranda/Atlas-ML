import { readPlotColors } from './canvas';
import { createHeroRun, drawHero, heroDuration, heroFinalPose, heroPose, type HeroPose } from './heroMotion';

const FRAME_MS = 1000 / 30;
const MAX_RATIO = 2;
const PHONE_WIDTH = 640;

function start(panel: HTMLElement, canvas: HTMLCanvasElement, label: HTMLElement | null) {
	const ctx = canvas.getContext('2d');
	if (!ctx) return;

	const connection = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection;
	const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches || !!connection?.saveData;
	const points = window.innerWidth < PHONE_WIDTH ? 48 : 96;

	let seed = 3;
	let run = createHeroRun(seed, points);
	let colors = readPlotColors(panel);
	let width = 0;
	let height = 0;
	let seconds = 0;
	let onScreen = true;
	let lastFrame = 0;
	let looping = false;
	let shownStep = '';

	const describe = (pose: HeroPose) => {
		const last = run.steps.length - 1;
		const step = Math.min(pose.step + (pose.recolor > 0.5 ? 1 : 0), last);
		const text = step === last ? `Converged after ${last} ${last === 1 ? 'step' : 'steps'}` : `Step ${step} of ${last}`;
		if (label && text !== shownStep) label.textContent = shownStep = text;
	};

	const paint = () => {
		const pose = still ? heroFinalPose(run) : heroPose(run, seconds);
		drawHero(ctx, width, height, run, pose, colors);
		describe(pose);
	};

	const resize = () => {
		const ratio = Math.min(window.devicePixelRatio || 1, MAX_RATIO);
		({ width, height } = canvas.getBoundingClientRect());
		canvas.width = Math.round(width * ratio);
		canvas.height = Math.round(height * ratio);
		ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		paint();
	};

	const frame = (now: number) => {
		if (!onScreen || document.hidden) {
			looping = false;
			return;
		}
		requestAnimationFrame(frame);
		const elapsed = now - lastFrame;
		if (elapsed < FRAME_MS) return;
		lastFrame = now;
		seconds += Math.min(elapsed, 100) / 1000;
		if (seconds > heroDuration(run)) {
			seconds = 0;
			seed += 1;
			run = createHeroRun(seed, points);
		}
		paint();
	};

	const resume = () => {
		if (still || looping || !onScreen || document.hidden) return;
		looping = true;
		lastFrame = performance.now();
		requestAnimationFrame(frame);
	};

	new ResizeObserver(resize).observe(canvas);
	new MutationObserver(() => {
		colors = readPlotColors(panel);
		paint();
	}).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
	new IntersectionObserver(([entry]) => {
		onScreen = entry.isIntersecting;
		resume();
	}).observe(canvas);
	document.addEventListener('visibilitychange', resume);
}

const panel = document.querySelector<HTMLElement>('[data-hero-live]');
const canvas = panel?.querySelector('canvas');
if (panel && canvas) {
	// The page's content comes first, so this waits until the browser has a moment.
	const idle = window.requestIdleCallback ?? ((run: () => void) => setTimeout(run, 200));
	idle(() => start(panel, canvas, panel.querySelector<HTMLElement>('[data-hero-step]')));
}
