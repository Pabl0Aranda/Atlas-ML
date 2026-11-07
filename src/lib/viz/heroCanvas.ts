import { readPlotColors } from './canvas';
import { createHeroRun, drawHero, heroDuration, heroFinalPose, heroPose } from './heroMotion';

const FRAME_MS = 1000 / 30;
const MAX_RATIO = 1.5;
const PHONE_WIDTH = 640;

function start(hero: HTMLElement, panel: HTMLElement) {
	const canvas = document.createElement('canvas');
	const ctx = canvas.getContext('2d');
	if (!ctx) return;
	canvas.className = 'hero-canvas';
	canvas.setAttribute('aria-hidden', 'true');
	panel.classList.add('has-live-hero');
	panel.prepend(canvas);

	const connection = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection;
	const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches || !!connection?.saveData;
	const points = window.innerWidth < PHONE_WIDTH ? 48 : 96;

	let seed = 3;
	let run = createHeroRun(seed, points);
	let colors = readPlotColors(hero);
	let width = 0;
	let height = 0;
	let seconds = 0;
	let onScreen = true;
	let lastFrame = 0;
	let looping = false;

	const paint = () => drawHero(ctx, width, height, run, still ? heroFinalPose(run) : heroPose(run, seconds), colors);

	const resize = () => {
		const ratio = Math.min(window.devicePixelRatio || 1, MAX_RATIO);
		width = panel.clientWidth;
		height = hero.getBoundingClientRect().bottom - panel.getBoundingClientRect().top;
		canvas.style.height = `${height}px`;
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

	new ResizeObserver(resize).observe(hero);
	new MutationObserver(() => {
		colors = readPlotColors(hero);
		paint();
	}).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
	new IntersectionObserver(([entry]) => {
		onScreen = entry.isIntersecting;
		resume();
	}).observe(canvas);
	document.addEventListener('visibilitychange', resume);
}

const hero = document.querySelector<HTMLElement>('.hero');
const panel = hero?.closest<HTMLElement>('.content-panel');
if (hero && panel) {
	// The page's content comes first, so this waits until the browser has a moment.
	const idle = window.requestIdleCallback ?? ((run: () => void) => setTimeout(run, 200));
	idle(() => start(hero, panel));
}
