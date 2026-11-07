import { describe, expect, it } from 'vitest';
import { FAMILIES } from '../../data/catalog';
import type { PlotColors } from './canvas';
import { createCardMotion, motionSeed } from './cardMotion';

const colors: PlotColors = {
	series: ['#111111', '#222222', '#333333', '#444444'],
	ink: '#ffffff',
	muted: '#888888',
	surface: '#000000',
	grid: '#555555',
};

/** A stand-in context that records every call and every style it is given. */
function recordingContext() {
	const calls: unknown[][] = [];
	const ctx = new Proxy(
		{},
		{
			get: (_, name) =>
				(...args: unknown[]) => {
					calls.push([name, ...args]);
				},
			set: (_, name, value) => {
				calls.push([name, value]);
				return true;
			},
		},
	) as unknown as CanvasRenderingContext2D;
	return { ctx, calls };
}

const TIMES = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1];
const SHAPES = [
	{ w: 400, h: 240 },
	{ w: 128, h: 128 },
];

describe('card motions', () => {
	for (const family of FAMILIES) {
		for (const { w, h } of SHAPES) {
			it(`${family} only draws finite values at ${w}x${h}`, () => {
				const motion = createCardMotion(family, 7, w / h);
				for (const t of TIMES) {
					const { ctx, calls } = recordingContext();
					motion(ctx, w, h, t, colors);
					expect(calls.length).toBeGreaterThan(2);
					for (const call of calls) {
						for (const arg of call) {
							if (typeof arg === 'number') expect(Number.isFinite(arg), `${String(call[0])} at t=${t}`).toBe(true);
							else expect(arg, `${String(call[0])} at t=${t}`).not.toBeUndefined();
						}
					}
				}
			});
		}

		it(`${family} changes over time and repeats for a seed`, () => {
			const frame = (seed: number, t: number) => {
				const { ctx, calls } = recordingContext();
				createCardMotion(family, seed, 5 / 3)(ctx, 400, 240, t, colors);
				return JSON.stringify(calls);
			};
			expect(frame(7, 0.1)).not.toBe(frame(7, 0.9));
			expect(frame(7, 0.5)).toBe(frame(7, 0.5));
		});
	}

	it('holds the last frame past the end', () => {
		const frame = (t: number) => {
			const { ctx, calls } = recordingContext();
			createCardMotion('Deep Learning', 3, 1)(ctx, 200, 200, t, colors);
			return JSON.stringify(calls);
		};
		expect(frame(1.4)).toBe(frame(1));
	});

	it('gives different cards different seeds', () => {
		expect(motionSeed('/algorithms/knn/')).not.toBe(motionSeed('/algorithms/svm/'));
	});
});
