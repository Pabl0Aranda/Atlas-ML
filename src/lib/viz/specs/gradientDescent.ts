import type { Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec, SceneLine } from '../../../types/viz';
import { descentStep, gradientSize, lowestLoss, startDescent, SURFACES, type DescentState, type Surface } from '../../ml/gradientDescent';
import { lines, num, vector } from '../equation';
import { contour, fixed } from '../scene';

/** Long enough for a slow but steady run to settle, short enough to stop a stalled one. */
export const MAX_STEPS = 200;
/** A gradient this small means the run has reached a flat spot. */
export const FLAT = 1e-3;
const LEVELS = 10;

export const SURFACE_NAMES = ['Bowl', 'Narrow valley', 'Two valleys', 'Curved valley'];
const surfaceOf = (p: Hyperparameters): Surface => SURFACES[p.surface] ?? SURFACES[0];

/**
 * Contour lines, each at half the height of the one above it, from the highest corner of
 * the plot down toward the floor. Halving keeps lines spread over the whole plot even when
 * the loss climbs steeply on one side.
 */
const contourCache = new Map<number, SceneLine[]>();
function contoursOf(index: number): SceneLine[] {
	const cached = contourCache.get(index);
	if (cached) return cached;
	const surface = SURFACES[index] ?? SURFACES[0];
	const floor = lowestLoss(surface);
	const top = Math.max(...[-1, 1].flatMap((x) => [-1, 1].map((y) => surface.loss(x, y))));
	const lines = Array.from({ length: LEVELS }, (_, k) => floor + (top - floor) / 2 ** (k + 1)).flatMap((level) =>
		contour(surface.loss, level, { tone: 'muted' }, 64),
	);
	contourCache.set(index, lines);
	return lines;
}

const isFlat = (s: DescentState, p: Hyperparameters) => gradientSize(surfaceOf(p), s.x, s.y) < FLAT;

/** Whether the last step turned back against the one before: the weights overshot and bounced. */
function zigzagging(s: DescentState): boolean {
	if (s.path.length < 3) return false;
	const [[ax, ay], [bx, by], [cx, cy]] = s.path.slice(-3);
	return (bx - ax) * (cx - bx) + (by - ay) * (cy - by) < 0;
}

export const gradientDescent: AlgorithmSpec<DescentState> = {
	datasets: [],
	editable: false,
	params: [
		{
			key: 'surface',
			label: 'Loss surface',
			value: 1,
			options: SURFACE_NAMES.map((label, value) => ({ label, value })),
		},
		{ key: 'alpha', label: 'Learning rate α', value: 0.1, min: 0.005, max: 0.5, step: 0.005 },
		{ key: 'beta', label: 'Momentum β', value: 0, min: 0, max: 0.95, step: 0.05 },
		{ key: 'startX', label: 'Start w₁', value: -0.8, min: -0.95, max: 0.95, step: 0.05 },
		{ key: 'startY', label: 'Start w₂', value: 0.6, min: -0.95, max: 0.95, step: 0.05 },
	],
	labeled: false,
	speed: 1,
	thumbnailSteps: 60,
	legend: () => ['Path of the weights'],
	init: (_, p) => startDescent(surfaceOf(p), p.startX, p.startY),
	step: (s, _, p) => descentStep(s, surfaceOf(p), p.alpha, p.beta),
	isDone: (s, p) => s.diverged || s.step >= MAX_STEPS || isFlat(s, p),
	scene(s, _, p) {
		return {
			points: s.path.map(([x, y]) => ({ x, y, cls: 0, scale: 0.5 })),
			lines: [...contoursOf(p.surface), ...(s.path.length > 1 ? [{ path: s.path, tone: 'ink' as const }] : [])],
			markers: [{ x: s.x, y: s.y, cls: 0 }],
		};
	},
	metrics(s, _, p) {
		return [
			{ label: 'Step', value: String(s.step) },
			{ label: 'Loss', value: s.diverged ? 'diverged' : fixed(s.losses.at(-1)!) },
			{ label: 'Gradient size', value: s.diverged ? '–' : fixed(gradientSize(surfaceOf(p), s.x, s.y)) },
			{ label: 'Weights', value: `(${s.x.toFixed(2)}, ${s.y.toFixed(2)})` },
		];
	},
	caption(s, _, p) {
		if (s.step === 0) return 'The weights start at the dot. Each step moves them against the gradient, the steepest way downhill. Press Step or Play.';
		if (s.diverged) return 'Diverged: each step overshot the valley by more than the last, so the weights flew off the plot. Lower the learning rate.';
		if (isFlat(s, p)) return `Converged after ${s.step} steps: the gradient is almost zero, so there is nowhere downhill left to go.`;
		if (s.step >= MAX_STEPS) return `Stopped after ${MAX_STEPS} steps without settling. A larger learning rate or some momentum would get further.`;
		if (zigzagging(s)) return 'Zigzag: the last step turned back on the one before. The step overshot the valley floor and bounced across it.';
		return 'The loss falls on every step. The steps shrink as the slope flattens near the bottom.';
	},
	series: (s) => ({ label: 'Loss per step', values: s.losses }),
	equation(s, _, p) {
		if (s.diverged) return '\\text{Diverged: the weights left the plot.}';
		const [gx, gy] = surfaceOf(p).gradient(s.x, s.y);
		const vx = p.beta * s.vx - p.alpha * gx;
		const vy = p.beta * s.vy - p.alpha * gy;
		// The rule shows its numbers, and only the results are written out as vectors, so it fits beside the plot.
		const gradient = `\\nabla L(w) = ${vector(gx, gy)}`;
		const next = vector(s.x + vx, s.y + vy);
		if (!p.beta) return lines(gradient, `w \\leftarrow w - ${num(p.alpha, 3)}\\,\\nabla L(w) = ${next}`);
		return lines(
			gradient,
			`v \\leftarrow ${num(p.beta)}\\,v - ${num(p.alpha, 3)}\\,\\nabla L = ${vector(vx, vy)}`,
			`w \\leftarrow w + v = ${next}`,
		);
	},
	describe(_, probe, p) {
		return `loss ${fixed(surfaceOf(p).loss(probe.x, probe.y))}`;
	},
};
