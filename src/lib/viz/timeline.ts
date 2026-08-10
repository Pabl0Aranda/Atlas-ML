/** Enough to scrub smoothly, while bounding memory on long runs. */
const MAX_FRAMES = 240;

/** Recorded states of a run, and the one on screen. */
export interface Timeline<S> {
	frames: S[];
	index: number;
}

export const createTimeline = <S>(state: S): Timeline<S> => ({ frames: [state], index: 0 });

export const currentFrame = <S>(timeline: Timeline<S>): S => timeline.frames[timeline.index];

export function seekFrame<S>(timeline: Timeline<S>, index: number): Timeline<S> {
	return { ...timeline, index: Math.min(timeline.frames.length - 1, Math.max(0, index)) };
}

/** Records a new state after the current frame, which discards any frames rewound past. */
export function pushFrame<S>(timeline: Timeline<S>, state: S, limit = MAX_FRAMES): Timeline<S> {
	const kept = timeline.frames.slice(0, timeline.index + 1);
	// Dropping every other frame keeps coverage of the whole run, at a coarser resolution.
	const frames = [...(kept.length >= limit ? kept.filter((_, i) => i % 2 === 0) : kept), state];
	return { frames, index: frames.length - 1 };
}
