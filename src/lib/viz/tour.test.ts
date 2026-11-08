import { describe, expect, it } from 'vitest';
import { TOUR_DONE, TOUR_START, TOUR_STEPS, tourReducer, type TourEvent } from './tour';

const run = (...events: TourEvent[]) => events.reduce(tourReducer, TOUR_START);

describe('home page tour', () => {
	it('has three steps, each done by a different sandbox action', () => {
		expect(TOUR_STEPS).toHaveLength(3);
		expect(new Set(TOUR_STEPS.map((s) => s.action)).size).toBe(3);
	});

	it('advances when the visitor does each step, and finishes after the last', () => {
		expect(run('plot')).toEqual({ step: 1, done: false });
		expect(run('plot', 'run')).toEqual({ step: 2, done: false });
		expect(run('plot', 'run', 'rewind')).toEqual(TOUR_DONE);
	});

	it('ignores actions meant for a later step, so nothing gets skipped', () => {
		expect(run('run', 'rewind')).toEqual(TOUR_START);
		expect(run('plot', 'rewind')).toEqual({ step: 1, done: false });
	});

	it('repeating a finished step does not jump ahead', () => {
		expect(run('plot', 'plot', 'plot')).toEqual({ step: 1, done: false });
	});

	it('offers Next, Back and Skip for visitors who would rather read', () => {
		expect(run('next', 'next')).toEqual({ step: 2, done: false });
		expect(run('next', 'back', 'back')).toEqual(TOUR_START);
		expect(run('next', 'skip')).toEqual(TOUR_DONE);
		expect(run('next', 'next', 'next')).toEqual(TOUR_DONE);
	});

	it('stays finished until restarted', () => {
		expect(run('skip', 'plot', 'back')).toEqual(TOUR_DONE);
		expect(run('skip', 'restart')).toEqual(TOUR_START);
	});
});
