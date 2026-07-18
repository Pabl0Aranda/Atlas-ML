import type { Scene, SceneGrid } from '../../types/viz';

const SIZE = 200;
const REGION_CELLS = 40;

const px = (x: number) => (((x + 1) / 2) * SIZE).toFixed(1);
const py = (y: number) => (((1 - y) / 2) * SIZE).toFixed(1);
const fill = (cls: number) => `var(--viz-series-${(cls % 4) + 1})`;

function regionRects(scene: Scene): string {
	if (!scene.region) return '';
	const cell = SIZE / REGION_CELLS;
	let rects = '';
	for (let r = 0; r < REGION_CELLS; r++) {
		let start = 0;
		const y = 1 - ((r + 0.5) / REGION_CELLS) * 2;
		const classAt = (c: number) => scene.region!(((c + 0.5) / REGION_CELLS) * 2 - 1, y);
		let current = classAt(0);
		for (let c = 1; c <= REGION_CELLS; c++) {
			const next = c < REGION_CELLS ? classAt(c) : -1;
			if (next === current) continue;
			// Runs of equal class merge into one rect to keep the markup small.
			rects += `<rect x="${start * cell}" y="${r * cell}" width="${(c - start) * cell}" height="${cell + 0.3}" fill="${fill(current)}"/>`;
			start = c;
			current = next;
		}
	}
	return `<g opacity=".18">${rects}</g>`;
}

function gridToSvg(grid: SceneGrid, label: string): string {
	const cell = 20;
	const center = (i: number) => [((i % grid.cols) + 0.5) * cell, (Math.floor(i / grid.cols) + 0.5) * cell];
	const rects = grid.cells
		.map((c, i) => {
			const [x, y] = center(i);
			const color = c.tile === 'cliff' ? fill(1) : c.tile === 'wall' ? 'var(--viz-muted)' : fill(0);
			const opacity = c.tile === 'cliff' || c.tile === 'wall' ? 1 : c.value * 0.6;
			return `<rect x="${x - cell / 2}" y="${y - cell / 2}" width="${cell}" height="${cell}" fill="${color}" fill-opacity="${opacity.toFixed(2)}" stroke="var(--viz-grid)"/>`;
		})
		.join('');
	const route = grid.path.map((i, n) => `${n ? 'L' : 'M'}${center(i).join(' ')}`).join('');
	return `<svg data-grid viewBox="0 0 ${grid.cols * cell} ${grid.rows * cell}" role="img" aria-label="${label}" style="background:var(--viz-surface)">${rects}<path d="${route}" fill="none" stroke="var(--viz-ink)" stroke-width="2"/></svg>`;
}

/** Static twin of the canvas renderer, for build-time thumbnails with no client JS. */
export function sceneToSvg(scene: Scene, label: string): string {
	if (scene.grid) return gridToSvg(scene.grid, label);
	const lines = (scene.lines ?? [])
		.filter((l) => l.tone === 'ink')
		.map((l) => {
			const d = l.path.map(([x, y], i) => `${i ? 'L' : 'M'}${px(x)} ${py(y)}`).join('');
			return `<path d="${d}" fill="none" stroke="var(--viz-ink)" stroke-width="2"${l.dashed ? ' stroke-dasharray="5 5"' : ''}/>`;
		})
		.join('');
	const points = scene.points
		.map((p) => `<circle cx="${px(p.x)}" cy="${py(p.y)}" r="${(3 * (p.scale ?? 1)).toFixed(1)}" fill="${p.muted ? 'var(--viz-muted)' : fill(p.cls)}" stroke="var(--viz-surface)"/>`)
		.join('');
	const markers = (scene.markers ?? [])
		.map((m) => `<circle cx="${px(m.x)}" cy="${py(m.y)}" r="7" fill="${fill(m.cls)}" stroke="var(--viz-ink)" stroke-width="2"/>`)
		.join('');
	return `<svg viewBox="0 0 ${SIZE} ${SIZE}" role="img" aria-label="${label}" style="background:var(--viz-surface)">${regionRects(scene)}${lines}${points}${markers}</svg>`;
}
