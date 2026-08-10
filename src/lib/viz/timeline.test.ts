import { describe, expect, it } from 'vitest';
import { createTimeline, currentFrame, pushFrame, seekFrame } from './timeline';

const filled = (count: number, limit?: number) => {
	let t = createTimeline(0);
	for (let i = 1; i < count; i++) t = pushFrame(t, i, limit);
	return t;
};

describe('timeline', () => {
	it('starts on its only frame', () => {
		expect(currentFrame(createTimeline('a'))).toBe('a');
	});

	it('moves to each frame it records', () => {
		const t = filled(4);
		expect(t.frames).toEqual([0, 1, 2, 3]);
		expect(currentFrame(t)).toBe(3);
	});

	it('rewinds without losing later frames', () => {
		const t = seekFrame(filled(4), 1);
		expect(currentFrame(t)).toBe(1);
		expect(t.frames).toHaveLength(4);
	});

	it('clamps seeking to the recorded range', () => {
		expect(seekFrame(filled(3), -5).index).toBe(0);
		expect(seekFrame(filled(3), 99).index).toBe(2);
	});

	it('branches from a rewound frame, dropping the old future', () => {
		const t = pushFrame(seekFrame(filled(4), 1), 9);
		expect(t.frames).toEqual([0, 1, 9]);
		expect(t.index).toBe(2);
	});

	it('stays bounded by thinning, while keeping the first and newest frames', () => {
		const t = filled(1000, 16);
		expect(t.frames.length).toBeLessThanOrEqual(16);
		expect(t.frames[0]).toBe(0);
		expect(currentFrame(t)).toBe(999);
	});
});
