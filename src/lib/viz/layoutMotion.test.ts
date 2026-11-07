import { describe, expect, it } from 'vitest';
import { planMotion, type Box } from './layoutMotion';

const VIEWPORT = 800;
const box = (top: number, left = 0): Box => ({ top, left, bottom: top + 100 });

describe('planMotion', () => {
	it('moves an element by the distance back to where it was', () => {
		expect(planMotion(box(300, 40), box(100, 10), VIEWPORT)).toEqual({ kind: 'move', dx: 30, dy: 200 });
	});

	it('leaves an element alone when it stays in place', () => {
		expect(planMotion(box(100), box(100.2), VIEWPORT)).toEqual({ kind: 'none' });
	});

	it('fades in an element that arrives from off screen', () => {
		expect(planMotion(box(2000), box(100), VIEWPORT)).toEqual({ kind: 'enter' });
	});

	it('fades out an element that leaves the screen', () => {
		expect(planMotion(box(100), box(2000), VIEWPORT)).toEqual({ kind: 'exit' });
	});

	it('skips a move that happens entirely off screen', () => {
		expect(planMotion(box(2000), box(1500), VIEWPORT)).toEqual({ kind: 'none' });
		expect(planMotion(box(-900), box(-500), VIEWPORT)).toEqual({ kind: 'none' });
	});

	it('fades in a newly shown element only when it can be seen', () => {
		expect(planMotion(null, box(100), VIEWPORT)).toEqual({ kind: 'enter' });
		expect(planMotion(null, box(2000), VIEWPORT)).toEqual({ kind: 'none' });
	});

	it('fades out a newly hidden element only when it could be seen', () => {
		expect(planMotion(box(100), null, VIEWPORT)).toEqual({ kind: 'exit' });
		expect(planMotion(box(2000), null, VIEWPORT)).toEqual({ kind: 'none' });
	});

	it('ignores an element that stays hidden', () => {
		expect(planMotion(null, null, VIEWPORT)).toEqual({ kind: 'none' });
	});
});
