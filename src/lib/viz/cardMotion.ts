import type { Family } from '../../data/catalog';
import { createRng } from '../ml/random';
import type { PlotColors } from './canvas';

/** Long enough to read the idea, short enough not to feel like a wait. */
export const CARD_MOTION_MS = 800;

/** Paints one frame. `t` runs from 0 to 1, and `w` and `h` are CSS pixels. */
export type CardMotion = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number, colors: PlotColors) => void;

type Rng = () => number;

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const span = (t: number, from: number, to: number) => clamp01((t - from) / (to - from));
const lerp = (a: number, b: number, p: number) => a + (b - a) * p;
const easeOut = (p: number) => 1 - (1 - p) ** 3;
const easeIn = (p: number) => p ** 3;
const easeInOut = (p: number) => (p < 0.5 ? 4 * p ** 3 : 1 - (-2 * p + 2) ** 3 / 2);

function dot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, alpha = 1) {
	ctx.globalAlpha = alpha;
	ctx.fillStyle = color;
	ctx.beginPath();
	ctx.arc(x, y, r, 0, Math.PI * 2);
	ctx.fill();
	ctx.globalAlpha = 1;
}

function ring(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, width: number, alpha = 1) {
	ctx.globalAlpha = alpha;
	ctx.strokeStyle = color;
	ctx.lineWidth = width;
	ctx.beginPath();
	ctx.arc(x, y, r, 0, Math.PI * 2);
	ctx.stroke();
	ctx.globalAlpha = 1;
}

function line(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, width: number, alpha = 1) {
	ctx.globalAlpha = alpha;
	ctx.strokeStyle = color;
	ctx.lineWidth = width;
	ctx.beginPath();
	ctx.moveTo(x1, y1);
	ctx.lineTo(x2, y2);
	ctx.stroke();
	ctx.globalAlpha = 1;
}

function block(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, color: string, alpha = 1) {
	ctx.globalAlpha = alpha;
	ctx.fillStyle = color;
	ctx.fillRect(x, y, w, h);
	ctx.globalAlpha = 1;
}

/** Supervised: a boundary searches for its angle, and points take the class of their side. */
function boundarySearch(rng: Rng): CardMotion {
	const target = rng() * Math.PI;
	const start = target + 1.5;
	const points = Array.from({ length: 36 }, () => ({ x: 0.06 + rng() * 0.88, y: 0.08 + rng() * 0.84 }));

	return (ctx, w, h, t, colors) => {
		const settle = easeInOut(span(t, 0, 0.78));
		const angle = lerp(start, target, settle) + Math.sin(t * 16) * 0.2 * (1 - settle);
		const reach = w + h;
		const shade = 0.16 * span(t, 0, 0.15);

		ctx.save();
		ctx.translate(w / 2, h / 2);
		ctx.rotate(angle);
		block(ctx, 0, -reach, reach, reach * 2, colors.series[0], shade);
		block(ctx, -reach, -reach, reach, reach * 2, colors.series[1], shade);
		ctx.restore();

		const side = (x: number, y: number) => (x - w / 2) * Math.cos(angle) + (y - h / 2) * Math.sin(angle);
		const r = Math.min(w, h) * 0.024;
		for (const p of points) {
			const x = p.x * w;
			const y = p.y * h;
			dot(ctx, x, y, r, colors.series[side(x, y) > 0 ? 0 : 1]);
		}

		const dx = -Math.sin(angle) * reach;
		const dy = Math.cos(angle) * reach;
		line(ctx, w / 2 - dx, h / 2 - dy, w / 2 + dx, h / 2 + dy, colors.ink, 2);
	};
}

/** Unsupervised: unlabeled points drift together, and the groups earn their colors. */
function clusterFormation(rng: Rng): CardMotion {
	const turn = rng() * Math.PI * 2;
	const centers = [0, 1, 2].map((k) => ({
		x: 0.5 + Math.cos(turn + (k * Math.PI * 2) / 3) * 0.27,
		y: 0.5 + Math.sin(turn + (k * Math.PI * 2) / 3) * 0.27,
	}));
	const points = Array.from({ length: 42 }, (_, i) => {
		const center = centers[i % 3];
		return {
			k: i % 3,
			fromX: rng(),
			fromY: rng(),
			toX: center.x + (rng() - 0.5) * 0.2,
			toY: center.y + (rng() - 0.5) * 0.2,
			delay: rng() * 0.25,
		};
	});

	return (ctx, w, h, t, colors) => {
		const r = Math.min(w, h) * 0.022;
		const pull = Math.sin(Math.PI * span(t, 0.15, 0.85)) * 0.3;
		for (const p of points) {
			const move = easeInOut(span(t, p.delay, p.delay + 0.6));
			const x = lerp(p.fromX, p.toX, move) * w;
			const y = lerp(p.fromY, p.toY, move) * h;
			line(ctx, x, y, centers[p.k].x * w, centers[p.k].y * h, colors.series[p.k], 1, pull);
			dot(ctx, x, y, r, colors.muted, 1 - move);
			dot(ctx, x, y, r, colors.series[p.k], move);
		}

		const appear = easeOut(span(t, 0.6, 0.85));
		centers.forEach((c, k) => {
			ring(ctx, c.x * w, c.y * h, Math.min(w, h) * 0.075 * appear, colors.series[k], 2, appear);
			dot(ctx, c.x * w, c.y * h, r * 1.3 * appear, colors.ink, appear);
		});
	};
}

/** Statistical: random samples rain down, pile up, and a density curve explains the pile. */
function samplesToDensity(rng: Rng): CardMotion {
	const BINS = 13;
	const middle = (BINS - 1) / 2 + (rng() - 0.5) * 1.5;
	const density = (bin: number) => Math.exp(-(((bin - middle) / 2.6) ** 2) / 2);
	const heights = Array.from({ length: BINS }, (_, i) => Math.round(density(i) * 8));
	const samples = heights.flatMap((height, bin) =>
		Array.from({ length: height }, (_, level) => ({ bin, level, delay: level * 0.05 + rng() * 0.28 })),
	);

	return (ctx, w, h, t, colors) => {
		const left = w * 0.08;
		const base = h * 0.9;
		const cell = (w * 0.84) / BINS;
		const rise = (h * 0.72) / 8;

		line(ctx, left, base, w - left, base, colors.muted, 1);
		for (const s of samples) {
			const fall = easeIn(span(t, s.delay, s.delay + 0.2));
			const landed = base - (s.level + 1) * rise;
			block(ctx, left + s.bin * cell + 1, lerp(-rise, landed, fall), cell - 2, rise - 1, colors.series[0], fall > 0 ? 0.85 : 0);
		}

		const drawn = easeInOut(span(t, 0.62, 0.96));
		if (drawn === 0) return;
		ctx.strokeStyle = colors.series[1];
		ctx.lineWidth = 2.5;
		ctx.beginPath();
		const steps = 48;
		for (let i = 0; i <= Math.round(steps * drawn); i++) {
			const bin = (i / steps) * BINS - 0.5;
			const x = left + (i / steps) * BINS * cell;
			const y = base - density(bin) * 8 * rise;
			if (i === 0) ctx.moveTo(x, y);
			else ctx.lineTo(x, y);
		}
		ctx.stroke();
	};
}

/** Reinforcement: an agent walks to the reward, and the reward's value flows back along the path. */
function agentReachesGoal(rng: Rng, aspect: number): CardMotion {
	const cols = aspect > 1.3 ? 7 : 5;
	const rows = aspect > 1.3 ? 4 : 5;
	const moves = [...Array<string>(cols - 1).fill('right'), ...Array<string>(rows - 1).fill('up')];
	for (let i = moves.length - 1; i > 0; i--) {
		const j = Math.floor(rng() * (i + 1));
		[moves[i], moves[j]] = [moves[j], moves[i]];
	}
	const path = [{ col: 0, row: rows - 1 }];
	for (const move of moves) {
		const last = path[path.length - 1];
		path.push(move === 'right' ? { col: last.col + 1, row: last.row } : { col: last.col, row: last.row - 1 });
	}
	const onPath = (col: number, row: number) => path.some((p) => p.col === col && p.row === row);
	const walls: { col: number; row: number }[] = [];
	for (let i = 0; i < 40 && walls.length < 4; i++) {
		const col = Math.floor(rng() * cols);
		const row = Math.floor(rng() * rows);
		if (!onPath(col, row)) walls.push({ col, row });
	}
	const goal = path[path.length - 1];

	return (ctx, w, h, t, colors) => {
		const size = Math.min((w * 0.9) / cols, (h * 0.9) / rows);
		const left = (w - size * cols) / 2;
		const top = (h - size * rows) / 2;
		const tile = (col: number, row: number, color: string, alpha: number) =>
			block(ctx, left + col * size + 1, top + row * size + 1, size - 2, size - 2, color, alpha);

		for (let col = 0; col < cols; col++) for (let row = 0; row < rows; row++) tile(col, row, colors.grid, 1);
		for (const wall of walls) tile(wall.col, wall.row, colors.muted, 0.7);
		tile(goal.col, goal.row, colors.series[3], 0.9);

		const walk = easeInOut(span(t, 0, 0.68)) * (path.length - 1);
		const step = Math.min(Math.floor(walk), path.length - 2);
		path.forEach((p, i) => {
			if (i < walk && i < path.length - 1) tile(p.col, p.row, colors.series[0], 0.3);
			const fromGoal = path.length - 1 - i;
			const value = span(t, 0.68 + fromGoal * 0.025, 0.76 + fromGoal * 0.025);
			if (i < path.length - 1) tile(p.col, p.row, colors.series[2], value * (0.85 - fromGoal * 0.05));
		});

		const a = path[step];
		const b = path[step + 1];
		const along = walk - step;
		const x = left + (lerp(a.col, b.col, along) + 0.5) * size;
		const y = top + (lerp(a.row, b.row, along) + 0.5) * size;
		dot(ctx, x, y, size * 0.28, colors.ink);

		const burst = span(t, 0.68, 1);
		if (burst > 0) ring(ctx, x, y, size * (0.4 + easeOut(burst) * 1.6), colors.series[3], 2.5, 1 - burst);
	};
}

/** Deep learning: a signal fires through the layers, and one output wins. */
function forwardPass(rng: Rng): CardMotion {
	const layers = [3, 5, 5, 2];
	const nodes = layers.map((count, layer) =>
		Array.from({ length: count }, (_, i) => ({ x: 0.13 + (layer / (layers.length - 1)) * 0.74, y: (i + 1) / (count + 1) })),
	);
	const edges = layers.slice(0, -1).flatMap((_, layer) =>
		nodes[layer].flatMap((from) =>
			nodes[layer + 1].map((to) => ({ layer, from, to, weight: rng(), delay: rng() * 0.06 })),
		),
	);
	const winner = rng() < 0.5 ? 0 : 1;
	const reachedAt = (layer: number) => layer * 0.24 + 0.04;

	return (ctx, w, h, t, colors) => {
		const r = Math.min(w, h) * 0.045;
		for (const e of edges) {
			const start = reachedAt(e.layer) + e.delay;
			const travel = span(t, start, start + 0.22);
			const x1 = e.from.x * w;
			const y1 = e.from.y * h;
			const x2 = e.to.x * w;
			const y2 = e.to.y * h;
			line(ctx, x1, y1, x2, y2, colors.muted, 0.6 + e.weight, 0.25);
			if (travel > 0) line(ctx, x1, y1, x2, y2, colors.series[0], 0.6 + e.weight, 0.5 * e.weight);
			if (travel > 0 && travel < 1) {
				const p = easeInOut(travel);
				dot(ctx, lerp(x1, x2, p), lerp(y1, y2, p), r * 0.38, colors.series[0]);
			}
		}

		nodes.forEach((column, layer) => {
			const lit = span(t, reachedAt(layer), reachedAt(layer) + 0.1);
			column.forEach((node, i) => {
				const isWinner = layer === layers.length - 1 && i === winner;
				const faded = layer === layers.length - 1 && !isWinner ? 0.35 : 1;
				dot(ctx, node.x * w, node.y * h, r, colors.surface);
				ring(ctx, node.x * w, node.y * h, r, colors.muted, 1.5);
				dot(ctx, node.x * w, node.y * h, r, isWinner ? colors.series[2] : colors.series[0], lit * faded);
			});
		});

		const pulse = span(t, 0.82, 1);
		const out = nodes[layers.length - 1][winner];
		if (pulse > 0) ring(ctx, out.x * w, out.y * h, r * (1 + easeOut(pulse) * 2), colors.series[2], 2, 1 - pulse);
	};
}

/** Data preparation: a sweep drops the outliers and files everything else into neat rows. */
function tidySweep(rng: Rng): CardMotion {
	const COLS = 8;
	const ROWS = 3;
	const clean = Array.from({ length: COLS * ROWS }, () => ({ x: 0.1 + rng() * 0.8, y: 0.1 + rng() * 0.8 }))
		.sort((a, b) => a.x - b.x)
		.map((p, i) => ({ ...p, toX: 0.14 + (Math.floor(i / ROWS) / (COLS - 1)) * 0.72, toY: 0.26 + ((i % ROWS) / (ROWS - 1)) * 0.48 }));
	const outliers = Array.from({ length: 6 }, () => ({ x: 0.08 + rng() * 0.84, y: rng() < 0.5 ? 0.06 + rng() * 0.1 : 0.84 + rng() * 0.1 }));

	return (ctx, w, h, t, colors) => {
		const r = Math.min(w, h) * 0.026;
		const sweep = lerp(-0.05, 1.05, easeInOut(span(t, 0.05, 0.65)));
		const passed = (x: number) => span(sweep, x, x + 0.04);
		const moveFor = (x: number) => lerp(0.05, 0.65, x) + 0.04;

		for (const p of clean) {
			const move = easeInOut(span(t, moveFor(p.x), moveFor(p.x) + 0.32));
			const x = lerp(p.x, p.toX, move) * w;
			const y = lerp(p.y, p.toY, move) * h;
			dot(ctx, x, y, r, colors.muted, 1 - move);
			dot(ctx, x, y, r, colors.series[2], move);
		}

		for (const o of outliers) {
			const gone = passed(o.x);
			const size = r * 1.1 * (1 - gone);
			const x = o.x * w;
			const y = o.y * h;
			if (size <= 0) continue;
			line(ctx, x - size, y - size, x + size, y + size, colors.series[1], 2);
			line(ctx, x - size, y + size, x + size, y - size, colors.series[1], 2);
		}

		if (sweep < 1) line(ctx, sweep * w, 0, sweep * w, h, colors.ink, 2, 0.8);
	};
}

/** Validation: k-fold rounds build up, with a different fold held out each time. */
function foldRotation(rng: Rng): CardMotion {
	const K = 5;
	const scores = Array.from({ length: K }, () => 0.7 + rng() * 0.3);

	return (ctx, w, h, t, colors) => {
		const left = w * 0.08;
		const top = h * 0.12;
		const cellW = (w * 0.7) / K;
		const cellH = (h * 0.76) / K;
		const scoreX = left + cellW * K + w * 0.03;
		const scoreW = w * 0.15;

		for (let round = 0; round < K; round++) {
			const start = 0.04 + round * 0.13;
			for (let fold = 0; fold < K; fold++) {
				const show = easeOut(span(t, start + fold * 0.015, start + fold * 0.015 + 0.1));
				const held = fold === round;
				const flash = held ? 0.35 * Math.sin(Math.PI * span(t, 0.8, 1)) : 0;
				const x = left + fold * cellW + 1 + (1 - show) * -8;
				block(ctx, x, top + round * cellH + 1, cellW - 2, cellH - 2, held ? colors.series[1] : colors.series[0], show * (held ? 0.65 + flash : 0.4));
			}
			const score = easeOut(span(t, start + 0.1, start + 0.2));
			block(ctx, scoreX, top + round * cellH + cellH * 0.3, scoreW * scores[round] * score, cellH * 0.4, colors.series[2], score);
		}
	};
}

/** Testing: predictions land in a confusion matrix, and the counts tick up. */
function confusionFill(rng: Rng): CardMotion {
	const correct = () => 34 + Math.floor(rng() * 12);
	const wrong = () => 2 + Math.floor(rng() * 7);
	const counts = [correct(), wrong(), wrong(), correct()];
	const drops = Array.from({ length: 28 }, () => {
		const cell = rng() < 0.84 ? (rng() < 0.5 ? 0 : 3) : rng() < 0.5 ? 1 : 2;
		return { cell, u: 0.15 + rng() * 0.7, v: 0.15 + rng() * 0.7, delay: rng() * 0.55 };
	});

	return (ctx, w, h, t, colors) => {
		const size = Math.min(w, h) * 0.86;
		const cell = size / 2;
		const left = (w - size) / 2;
		const top = (h - size) / 2;
		const fill = easeOut(span(t, 0.1, 0.8));
		const origin = (i: number) => ({ x: left + (i % 2) * cell, y: top + Math.floor(i / 2) * cell });

		counts.forEach((count, i) => {
			const { x, y } = origin(i);
			const diagonal = i === 0 || i === 3;
			block(ctx, x + 1, y + 1, cell - 2, cell - 2, colors.grid, 1);
			block(ctx, x + 1, y + 1, cell - 2, cell - 2, diagonal ? colors.series[2] : colors.series[1], fill * (diagonal ? 0.7 : 0.3));
		});

		const r = Math.min(w, h) * 0.018;
		for (const d of drops) {
			const fall = easeIn(span(t, d.delay, d.delay + 0.22));
			if (fall === 0 || fall === 1) continue;
			const { x, y } = origin(d.cell);
			dot(ctx, x + d.u * cell, lerp(-r, y + d.v * cell, fall), r, colors.ink, 0.9);
		}

		ctx.fillStyle = colors.ink;
		ctx.font = `600 ${Math.round(cell * 0.34)}px ui-sans-serif, system-ui, sans-serif`;
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		counts.forEach((count, i) => {
			const { x, y } = origin(i);
			ctx.fillText(String(Math.round(count * fill)), x + cell / 2, y + cell / 2);
		});
	};
}

const MOTIONS: Record<Family, (rng: Rng, aspect: number) => CardMotion> = {
	'Data Preparation': tidySweep,
	Validation: foldRotation,
	Testing: confusionFill,
	'Supervised Learning': boundarySearch,
	'Unsupervised Learning': clusterFormation,
	'Statistical Learning': samplesToDensity,
	'Reinforcement Learning': agentReachesGoal,
	'Deep Learning': forwardPass,
};

/** Turns a card's name into a seed, so each card plays its own version every time. */
export function motionSeed(text: string): number {
	let hash = 2166136261;
	for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
	return hash >>> 0;
}

export function createCardMotion(family: Family, seed: number, aspect: number): CardMotion {
	const paint = MOTIONS[family](createRng(seed), aspect);
	return (ctx, w, h, t, colors) => {
		ctx.fillStyle = colors.surface;
		ctx.fillRect(0, 0, w, h);
		paint(ctx, w, h, clamp01(t), colors);
	};
}
