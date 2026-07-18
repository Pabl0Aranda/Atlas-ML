import type { Scene, SceneGrid } from '../../types/viz';

export interface PlotColors {
	series: string[];
	ink: string;
	muted: string;
	surface: string;
	grid: string;
}

const REGION_RESOLUTION = 72;
const REGION_ALPHA = 42;

export function readPlotColors(element: Element): PlotColors {
	const css = getComputedStyle(element);
	const get = (name: string) => css.getPropertyValue(name).trim();
	return {
		series: [1, 2, 3, 4].map((i) => get(`--viz-series-${i}`)),
		ink: get('--viz-ink'),
		muted: get('--viz-muted'),
		surface: get('--viz-surface'),
		grid: get('--viz-grid'),
	};
}

const hexToRgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

function drawRegion(ctx: CanvasRenderingContext2D, size: number, scene: Scene, colors: PlotColors) {
	if (!scene.region) return;
	const n = REGION_RESOLUTION;
	const image = new ImageData(n, n);
	const rgb = colors.series.map(hexToRgb);
	for (let r = 0; r < n; r++) {
		for (let c = 0; c < n; c++) {
			const cls = scene.region(((c + 0.5) / n) * 2 - 1, 1 - ((r + 0.5) / n) * 2);
			const [red, green, blue] = rgb[cls % rgb.length];
			image.data.set([red, green, blue, REGION_ALPHA], (r * n + c) * 4);
		}
	}
	const buffer = document.createElement('canvas');
	buffer.width = buffer.height = n;
	buffer.getContext('2d')!.putImageData(image, 0, 0);
	ctx.imageSmoothingEnabled = false;
	ctx.drawImage(buffer, 0, 0, size, size);
}

/** Shape doubles the class color, so identity never depends on color alone. */
function drawMark(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, cls: number) {
	ctx.beginPath();
	switch (cls % 4) {
		case 0:
			ctx.arc(x, y, r, 0, Math.PI * 2);
			break;
		case 1:
			ctx.rect(x - r * 0.88, y - r * 0.88, r * 1.76, r * 1.76);
			break;
		case 2:
			ctx.moveTo(x, y - r * 1.2);
			ctx.lineTo(x + r * 1.1, y + r * 0.85);
			ctx.lineTo(x - r * 1.1, y + r * 0.85);
			ctx.closePath();
			break;
		default:
			ctx.moveTo(x, y - r * 1.25);
			ctx.lineTo(x + r * 1.25, y);
			ctx.lineTo(x, y + r * 1.25);
			ctx.lineTo(x - r * 1.25, y);
			ctx.closePath();
	}
}

/** Draws a gridworld across the full canvas: value shading, policy arrows, route and agent. */
export function drawGrid(ctx: CanvasRenderingContext2D, width: number, grid: SceneGrid, colors: PlotColors) {
	const cell = width / grid.cols;
	const center = (i: number): [number, number] => [
		((i % grid.cols) + 0.5) * cell,
		(Math.floor(i / grid.cols) + 0.5) * cell,
	];
	ctx.fillStyle = colors.surface;
	ctx.fillRect(0, 0, width, cell * grid.rows);
	ctx.textAlign = 'center';
	ctx.textBaseline = 'middle';
	ctx.font = `600 ${Math.round(cell * 0.38)}px sans-serif`;

	grid.cells.forEach((c, i) => {
		const [x, y] = center(i);
		const left = x - cell / 2;
		const top = y - cell / 2;
		if (c.tile === 'wall') ctx.fillStyle = colors.muted;
		else if (c.tile === 'cliff') ctx.fillStyle = colors.series[1];
		else ctx.fillStyle = colors.series[0];
		ctx.globalAlpha = c.tile === 'wall' || c.tile === 'cliff' ? 1 : c.value * 0.6;
		ctx.fillRect(left, top, cell, cell);
		ctx.globalAlpha = 1;
		ctx.strokeStyle = colors.grid;
		ctx.lineWidth = 1;
		ctx.strokeRect(left, top, cell, cell);

		const glyph = c.tile === 'start' ? 'S' : c.tile === 'goal' ? 'G' : c.tile === 'cliff' ? '×' : null;
		if (glyph) {
			ctx.fillStyle = c.tile === 'cliff' ? colors.surface : colors.ink;
			ctx.fillText(glyph, x, y);
		} else if (c.arrow !== undefined) {
			const [dx, dy] = [[0, -1], [1, 0], [0, 1], [-1, 0]][c.arrow];
			const r = cell * 0.2;
			ctx.strokeStyle = colors.muted;
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.moveTo(x - dx * r, y - dy * r);
			ctx.lineTo(x + dx * r, y + dy * r);
			ctx.lineTo(x + dx * r * 0.2 - dy * r * 0.6, y + dy * r * 0.2 + dx * r * 0.6);
			ctx.moveTo(x + dx * r, y + dy * r);
			ctx.lineTo(x + dx * r * 0.2 + dy * r * 0.6, y + dy * r * 0.2 - dx * r * 0.6);
			ctx.stroke();
		}
	});

	if (grid.path.length > 1) {
		ctx.strokeStyle = colors.ink;
		ctx.lineWidth = 2;
		ctx.lineJoin = 'round';
		ctx.beginPath();
		grid.path.forEach((i, n) => (n ? ctx.lineTo(...center(i)) : ctx.moveTo(...center(i))));
		ctx.stroke();
	}

	const [ax, ay] = center(grid.agent);
	ctx.beginPath();
	ctx.arc(ax, ay, cell * 0.26, 0, Math.PI * 2);
	ctx.fillStyle = colors.ink;
	ctx.strokeStyle = colors.surface;
	ctx.lineWidth = 2;
	ctx.fill();
	ctx.stroke();
}

export function drawScene(ctx: CanvasRenderingContext2D, size: number, scene: Scene, colors: PlotColors) {
	const px = (x: number) => ((x + 1) / 2) * size;
	const py = (y: number) => ((1 - y) / 2) * size;

	ctx.fillStyle = colors.surface;
	ctx.fillRect(0, 0, size, size);
	drawRegion(ctx, size, scene, colors);

	ctx.strokeStyle = colors.grid;
	ctx.lineWidth = 1;
	ctx.setLineDash([]);
	for (const t of [-0.5, 0, 0.5]) {
		ctx.beginPath();
		ctx.moveTo(px(t), 0);
		ctx.lineTo(px(t), size);
		ctx.moveTo(0, py(t));
		ctx.lineTo(size, py(t));
		ctx.stroke();
	}

	ctx.lineCap = 'round';
	ctx.lineJoin = 'round';
	for (const line of (scene.lines ?? []).filter((l) => l.tone === 'muted').concat((scene.lines ?? []).filter((l) => l.tone === 'ink'))) {
		ctx.strokeStyle = line.tone === 'ink' ? colors.ink : colors.muted;
		ctx.lineWidth = line.tone === 'ink' ? 2 : 1;
		ctx.setLineDash(line.dashed ? [5, 5] : []);
		ctx.beginPath();
		line.path.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
		ctx.stroke();
	}
	ctx.setLineDash([]);

	const radius = Math.max(3.5, size / 130);
	for (const p of scene.points) {
		const x = px(p.x);
		const y = py(p.y);
		const r = radius * (p.scale ?? 1);
		if (p.ring) {
			ctx.beginPath();
			ctx.arc(x, y, r + 4, 0, Math.PI * 2);
			ctx.strokeStyle = colors.ink;
			ctx.lineWidth = 1.5;
			ctx.stroke();
		}
		drawMark(ctx, x, y, r, p.cls);
		ctx.fillStyle = p.muted ? colors.muted : colors.series[p.cls % colors.series.length];
		ctx.strokeStyle = colors.surface;
		ctx.lineWidth = 1.5;
		ctx.fill();
		ctx.stroke();
	}

	for (const m of scene.markers ?? []) {
		drawMark(ctx, px(m.x), py(m.y), radius * 2.1, m.cls);
		ctx.fillStyle = colors.series[m.cls % colors.series.length];
		ctx.strokeStyle = colors.ink;
		ctx.lineWidth = 2;
		ctx.fill();
		ctx.stroke();
	}
}
