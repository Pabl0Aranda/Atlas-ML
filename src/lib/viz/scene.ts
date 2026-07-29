import type { Dataset, Hyperparameters } from '../../types/ml';
import type { AlgorithmSpec, Scene, SceneLine, ScenePoint } from '../../types/viz';

export const classPoints = (data: Dataset): ScenePoint[] => data.map((p) => ({ x: p.x, y: p.y, cls: p.label }));

export const defaultParams = (spec: AlgorithmSpec): Hyperparameters =>
	Object.fromEntries(spec.params.map((p) => [p.key, p.value]));

export const fixed = (value: number, digits = 3): string =>
	Number.isFinite(value) ? value.toFixed(digits) : 'diverged';

export const percent = (value: number): string => `${(value * 100).toFixed(1)}%`;

/** A count with its noun, such as "1 cluster" or "3 clusters". */
export const counted = (count: number, noun: string): string => `${count} ${noun}${count === 1 ? '' : 's'}`;

/** Marching squares: traces where f crosses level as short segments. */
export function contour(
	f: (x: number, y: number) => number,
	level: number,
	style: Omit<SceneLine, 'path'>,
	resolution = 48,
): SceneLine[] {
	const size = 2 / resolution;
	const values: number[][] = [];
	for (let r = 0; r <= resolution; r++) {
		values.push([]);
		for (let c = 0; c <= resolution; c++) values[r].push(f(-1 + c * size, -1 + r * size) - level);
	}
	const lines: SceneLine[] = [];
	const crossing = (ax: number, ay: number, av: number, bx: number, by: number, bv: number): [number, number] => {
		const t = av / (av - bv);
		return [ax + t * (bx - ax), ay + t * (by - ay)];
	};
	for (let r = 0; r < resolution; r++) {
		for (let c = 0; c < resolution; c++) {
			const x = -1 + c * size;
			const y = -1 + r * size;
			const [a, b, d, e] = [values[r][c], values[r][c + 1], values[r + 1][c + 1], values[r + 1][c]];
			const hits: [number, number][] = [];
			if (a > 0 !== b > 0) hits.push(crossing(x, y, a, x + size, y, b));
			if (b > 0 !== d > 0) hits.push(crossing(x + size, y, b, x + size, y + size, d));
			if (d > 0 !== e > 0) hits.push(crossing(x + size, y + size, d, x, y + size, e));
			if (e > 0 !== a > 0) hits.push(crossing(x, y + size, e, x, y, a));
			for (let i = 0; i + 1 < hits.length; i += 2) lines.push({ ...style, path: [hits[i], hits[i + 1]] });
		}
	}
	return lines;
}

// Each card appears on several pages, and slow specs should only train once per build.
const thumbnails = new WeakMap<AlgorithmSpec, Scene>();

/** Runs a spec for its thumbnail step count and returns the final scene. */
export function thumbnailScene(spec: AlgorithmSpec, data: Dataset): Scene {
	const cached = thumbnails.get(spec);
	if (cached) return cached;
	const params = defaultParams(spec);
	let state = spec.init(data, params);
	for (let i = 0; i < spec.thumbnailSteps; i++) state = spec.step(state, data, params);
	const scene = spec.scene(state, data, params);
	thumbnails.set(spec, scene);
	return scene;
}
