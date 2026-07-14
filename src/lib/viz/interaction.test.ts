import { describe, expect, it } from 'vitest';
import { moveProbe, probeStep, toolOptions } from './interaction';

describe('toolOptions', () => {
	it('offers one tool per class on labeled specs, plus inspect', () => {
		expect(toolOptions(['Class A', 'Class B', 'Class C'], true)).toEqual([
			{ value: 0, label: 'Class A' },
			{ value: 1, label: 'Class B' },
			{ value: 2, label: 'Class C' },
			{ value: 'inspect', label: 'Inspect' },
		]);
	});

	it('ignores legend entries on unlabeled specs, which are not classes', () => {
		expect(toolOptions(['Kept', 'Flagged outlier'], false)).toEqual([
			{ value: 0, label: 'Sample' },
			{ value: 'inspect', label: 'Inspect' },
		]);
	});
});

describe('moveProbe', () => {
	const step = { x: 0.1, y: 0.2 };

	it('moves in plot coordinates, where up is positive y', () => {
		expect(moveProbe({ x: 0, y: 0 }, 'ArrowUp', step)).toEqual({ x: 0, y: 0.2 });
		expect(moveProbe({ x: 0, y: 0 }, 'ArrowDown', step)).toEqual({ x: 0, y: -0.2 });
		expect(moveProbe({ x: 0, y: 0 }, 'ArrowLeft', step)).toEqual({ x: -0.1, y: 0 });
		expect(moveProbe({ x: 0, y: 0 }, 'ArrowRight', step)).toEqual({ x: 0.1, y: 0 });
	});

	it('stays strictly inside the plot, so grid lookups cannot overflow', () => {
		expect(moveProbe({ x: 0.95, y: -0.95 }, 'ArrowRight', step)!.x).toBeLessThan(1);
		expect(moveProbe({ x: 0.95, y: -0.95 }, 'ArrowDown', step)!.y).toBeGreaterThan(-1);
	});

	it('returns null for keys that do not move the cursor', () => {
		expect(moveProbe({ x: 0, y: 0 }, 'Tab', step)).toBeNull();
	});
});

describe('probeStep', () => {
	it('moves a fixed distance on plots, four times further when coarse', () => {
		expect(probeStep(undefined, false)).toEqual({ x: 0.05, y: 0.05 });
		expect(probeStep(undefined, true)).toEqual({ x: 0.2, y: 0.2 });
	});

	it('moves one cell at a time on gridworlds', () => {
		expect(probeStep({ cols: 4, rows: 8 }, false)).toEqual({ x: 0.5, y: 0.25 });
		expect(probeStep({ cols: 4, rows: 8 }, true)).toEqual({ x: 0.5, y: 0.25 });
	});
});
