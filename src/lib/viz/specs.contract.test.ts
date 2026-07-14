import { describe, expect, it } from 'vitest';
import { generateDataset } from '../ml/datasets';
import { ALGORITHM_IDS, loadSpec } from './registry';
import { defaultParams } from './scene';
import type { Dataset, Hyperparameters } from '../../types/ml';
import type { AlgorithmSpec, Scene } from '../../types/viz';

/** Longer than any spec's default run; a spec that hits this never finishes. */
const MAX_STEPS = 100000;

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
