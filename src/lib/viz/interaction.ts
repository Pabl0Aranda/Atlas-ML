import type { SceneGrid } from '../../types/viz';

/** What a click, tap or Enter does on the plot: add a sample of a class, or only read the location. */
export type PlotTool = number | 'inspect';

export interface Point2D {
	x: number;
	y: number;
}

const KEY_STEP = 0.05;
const COARSE_FACTOR = 4;
/** Just inside the plot, because specs map 1 to one past the last grid cell. */
const EDGE = 0.999;

const DIRECTIONS: Record<string, Point2D> = {
	ArrowUp: { x: 0, y: 1 },
	ArrowDown: { x: 0, y: -1 },
	ArrowLeft: { x: -1, y: 0 },
	ArrowRight: { x: 1, y: 0 },
};

const clamp = (v: number) => Math.min(EDGE, Math.max(-EDGE, v));

export function toolOptions(legend: string[], labeled: boolean): { value: PlotTool; label: string }[] {
	const classes = labeled ? legend : ['Sample'];
	return [...classes.map((label, value) => ({ value, label })), { value: 'inspect', label: 'Inspect' }];
}

/** Arrow keys walk gridworlds cell by cell, and plots in small steps that are coarser with Shift. */
export function probeStep(grid: Pick<SceneGrid, 'cols' | 'rows'> | undefined, coarse: boolean): Point2D {
	if (grid) return { x: 2 / grid.cols, y: 2 / grid.rows };
	const size = coarse ? KEY_STEP * COARSE_FACTOR : KEY_STEP;
	return { x: size, y: size };
}

/** The probe after an arrow key, or null when the key is not an arrow. */
export function moveProbe(probe: Point2D, key: string, step: Point2D): Point2D | null {
	const direction = DIRECTIONS[key];
	if (!direction) return null;
	return { x: clamp(probe.x + direction.x * step.x), y: clamp(probe.y + direction.y * step.y) };
}
