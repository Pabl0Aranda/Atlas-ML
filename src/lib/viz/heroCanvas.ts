import { readPlotColors } from './canvas';
import { addHeroPoint, createHeroRun, drawHero, HERO_RESUME_AT, heroDuration, heroFinalPose, heroPose, toPlot, type HeroPose } from './heroMotion';
import { moveProbe, probeStep, type Point2D } from './interaction';

const FRAME_MS = 1000 / 30;
const MAX_RATIO = 2;
const PHONE_WIDTH = 640;

interface HeroParts {
	panel: HTMLElement;
	canvas: HTMLCanvasElement;
	label: HTMLElement | null;
	announce: HTMLElement | null;
	reset: HTMLButtonElement | null;
}

function start({ panel, canvas, label, announce, reset }: HeroParts) {
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
	/** Once a visitor adds a point, the run holds on its result instead of moving on to new data. */
	let handsOn = false;
	/** The keyboard cursor, in the sandboxes' orientation: y grows upward. Shown only while focused from the keyboard. */
	let cursor: Point2D = { x: 0, y: 0 };
	let showCursor = false;

	const holdAt = () => heroDuration(run) - HERO_RESUME_AT;
	const currentPose = (): HeroPose => (still ? heroFinalPose(run) : heroPose(run, seconds));
	const steps = () => run.steps.length - 1;
	const plural = (n: number) => `${n} ${n === 1 ? 'step' : 'steps'}`;

	const describe = (pose: HeroPose) => {
		const last = steps();
		const step = Math.min(pose.step + (pose.recolor > 0.5 ? 1 : 0), last);
		const text = step === last ? `Converged after ${plural(last)}` : `Step ${step} of ${last}`;
		if (label && text !== shownStep) label.textContent = shownStep = text;
	};

	const paint = () => {
		const pose = currentPose();
		drawHero(ctx, width, height, run, pose, colors, showCursor ? { x: cursor.x, y: -cursor.y } : undefined);
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
		const elapsed = now - lastFrame;
		if (elapsed < FRAME_MS) {
			requestAnimationFrame(frame);
			return;
		}
		lastFrame = now;
		seconds += Math.min(elapsed, 100) / 1000;
		if (handsOn && seconds >= holdAt()) {
			seconds = holdAt();
			looping = false;
			paint();
			return;
		}
		if (seconds > heroDuration(run)) {
			seconds = 0;
			seed += 1;
			run = createHeroRun(seed, points);
		}
		paint();
		requestAnimationFrame(frame);
	};

	const resume = () => {
		if (still || looping || !onScreen || document.hidden) return;
		looping = true;
		lastFrame = performance.now();
		requestAnimationFrame(frame);
	};

	const say = (text: string) => {
		if (announce) announce.textContent = text;
	};

	const addPoint = (point: Point2D) => {
		run = addHeroPoint(run, currentPose().centroids, point);
		seconds = HERO_RESUME_AT;
		handsOn = true;
		if (reset) reset.hidden = false;
		say(`Point added. k-means settles again after ${plural(steps())}. You have added ${run.added}.`);
		paint();
		resume();
	};

	canvas.addEventListener('click', (event) => {
		const box = canvas.getBoundingClientRect();
		addPoint(toPlot(event.clientX - box.left, event.clientY - box.top, box.width, box.height));
	});

	canvas.addEventListener('keydown', (event) => {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			showCursor = true;
			addPoint({ x: cursor.x, y: -cursor.y });
			return;
		}
		const next = moveProbe(cursor, event.key, probeStep(undefined, event.shiftKey));
		if (!next) return;
		event.preventDefault();
		cursor = next;
		showCursor = true;
		paint();
	});

	canvas.addEventListener('focus', () => {
		showCursor = canvas.matches(':focus-visible');
		paint();
	});
	canvas.addEventListener('blur', () => {
		showCursor = false;
		paint();
	});

	reset?.addEventListener('click', () => {
		handsOn = false;
		run = createHeroRun(seed, points);
		seconds = HERO_RESUME_AT;
		reset.hidden = true;
		say('Back to the original run.');
		canvas.focus({ preventScroll: true });
		paint();
		resume();
	});

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
	idle(() =>
		start({
			panel,
			canvas,
			label: panel.querySelector<HTMLElement>('[data-hero-step]'),
			announce: panel.querySelector<HTMLElement>('[data-hero-announce]'),
			reset: panel.querySelector<HTMLButtonElement>('[data-hero-reset]'),
		}),
	);
}
