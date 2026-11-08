import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { generateDataset } from '../ml/datasets';
import { PRESETS } from './presets';
import { ALGORITHM_IDS, loadSpec } from './registry';
import { composeData, defaultSetup, testAccuracy } from './sandboxSetup';
import { defaultParams } from './scene';
import type { Dataset, Hyperparameters } from '../../types/ml';
import type { AlgorithmSpec, Scene } from '../../types/viz';

/** Longer than any spec's default run; a spec that hits this never finishes. */
const MAX_STEPS = 100000;
/** Sandbox ticks to record, which reaches past the first update of the slowest learners. */
const FRAME_TICKS = 12;

const dataFor = (spec: AlgorithmSpec): Dataset => (spec.datasets.length ? generateDataset(spec.datasets[0]) : []);

const coordinates = (scene: Scene): number[] => [
	...scene.points.flatMap((p) => [p.x, p.y]),
	...(scene.markers ?? []).flatMap((m) => [m.x, m.y]),
	...(scene.lines ?? []).flatMap((l) => l.path.flat()),
];

function expectValidFrame(spec: AlgorithmSpec, state: unknown, data: Dataset, params: Hyperparameters) {
	const scene = spec.scene(state, data, params);
	expect(coordinates(scene).filter((v) => !Number.isFinite(v))).toEqual([]);
	if (scene.grid) expect(scene.grid.cells).toHaveLength(scene.grid.cols * scene.grid.rows);
	if (scene.region) expect(spec.legend(params)[scene.region(0.1, -0.2)]).toBeDefined();
	for (const metric of spec.metrics(state, data, params)) expect(metric.value).not.toMatch(/NaN|undefined/);
	expect(spec.caption(state, data, params)).not.toMatch(/NaN|undefined/);
	if (spec.equation) {
		const latex = spec.equation(state, data, params);
		expect(latex).not.toMatch(/NaN|undefined|infty/);
		expect(() => katex.renderToString(latex, { displayMode: true, throwOnError: true })).not.toThrow();
	}
}

describe.each(ALGORITHM_IDS)('%s spec', (id) => {
	it('produces valid frames from start to finish, and terminates', async () => {
		const spec = await loadSpec(id);
		const data = dataFor(spec);
		const params = defaultParams(spec);
		let state = spec.init(data, params);
		expectValidFrame(spec, state, data, params);
		let steps = 0;
		while (!spec.isDone(state, params) && steps < MAX_STEPS) {
			state = spec.step(state, data, params);
			steps++;
		}
		expect(steps).toBeLessThan(MAX_STEPS);
		expectValidFrame(spec, state, data, params);
	}, 120000);

	it('keeps recorded frames intact while the run continues', async () => {
		const spec = await loadSpec(id);
		const data = dataFor(spec);
		const params = defaultParams(spec);
		const view = (state: unknown) => {
			const scene = spec.scene(state, data, params);
			const probes = [-0.6, -0.2, 0.2, 0.6].flatMap((x) => [-0.6, 0.2].map((y) => scene.region?.(x, y)));
			return JSON.stringify([scene.points, scene.lines, scene.grid, probes, spec.metrics(state, data, params)]);
		};
		const copy = spec.snapshot ?? ((s: unknown) => s);
		let frame = spec.init(data, params);
		const recorded: [unknown, string][] = [];
		for (let tick = 0; tick < FRAME_TICKS && !spec.isDone(frame, params); tick++) {
			recorded.push([frame, view(frame)]);
			let next = copy(frame);
			for (let i = 0; i < spec.speed && !spec.isDone(next, params); i++) next = spec.step(next, data, params);
			frame = next;
		}
		for (const [state, seen] of recorded) expect(view(state)).toBe(seen);
	}, 120000);

	it('scores held-out samples only with a region that predicts labels', async () => {
		const spec = await loadSpec(id);
		if (!spec.holdout) return;
		expect(spec.labeled).toBe(true);
		const { train, test } = composeData({ ...defaultSetup(spec), holdout: true });
		const params = defaultParams(spec);
		let state = spec.init(train, params);
		for (let i = 0; i < 400 && !spec.isDone(state, params); i++) state = spec.step(state, train, params);
		expect(testAccuracy(spec.scene(state, train, params).region, test)).toBeGreaterThan(0.6);
	}, 120000);

	it('only offers presets the spec can apply', async () => {
		const spec = await loadSpec(id);
		for (const preset of PRESETS[id] ?? []) {
			if (preset.dataset) expect(spec.datasets).toContain(preset.dataset);
			if (preset.holdout) expect(spec.holdout).toBe(true);
			for (const [key, value] of Object.entries(preset.params ?? {})) {
				const param = spec.params.find((p) => p.key === key);
				expect(param, `${preset.label}: ${key}`).toBeDefined();
				if (param!.options) expect(param!.options.map((o) => o.value)).toContain(value);
				else {
					expect(value).toBeGreaterThanOrEqual(param!.min!);
					expect(value).toBeLessThanOrEqual(param!.max!);
				}
			}
		}
	});

	it('only marks a parameter live when the run does not depend on it', async () => {
		const spec = await loadSpec(id);
		const data = dataFor(spec);
		const params = defaultParams(spec);
		const trace = (p: Hyperparameters) => {
			const states = [spec.init(data, p)];
			for (let i = 0; i < 20 && !spec.isDone(states[i], p); i++) states.push(spec.step(states[i], data, p));
			return JSON.stringify(states);
		};
		for (const param of spec.params.filter((p) => p.live)) {
			const other = param.options ? param.options.find((o) => o.value !== param.value)!.value : param.max!;
			expect(trace({ ...params, [param.key]: other })).toBe(trace(params));
		}
	}, 120000);
});
