/** What a visitor did, as the home page tour sees it. The first three are actions in the sandbox itself. */
export type TourEvent = 'plot' | 'run' | 'rewind' | 'next' | 'back' | 'skip' | 'restart';

export interface TourStep {
	/** The sandbox action that completes this step. */
	action: 'plot' | 'run' | 'rewind';
	title: string;
	text: string;
	/** Which control to highlight. */
	target: 'plot' | 'run' | 'timeline';
}

export const TOUR_STEPS: TourStep[] = [
	{
		action: 'plot',
		title: 'Add a sample',
		text: 'Click or tap anywhere on the plot to add a point of the selected class. Switch the Plot tool to Class B to add the other class. From the keyboard, focus the plot, move with the arrow keys and press Enter.',
		target: 'plot',
	},
	{
		action: 'run',
		title: 'Run the algorithm',
		text: 'Press Step to move forward one step, or Play to watch it run. Here each step raises k, the number of neighbors that vote, so the boundary gets smoother.',
		target: 'run',
	},
	{
		action: 'rewind',
		title: 'Rewind',
		text: 'Drag the Timeline, or press Back, to return to any earlier moment. Stepping from there starts a new branch.',
		target: 'timeline',
	},
];

export interface TourState {
	/** Index of the current step. */
	step: number;
	done: boolean;
}

export const TOUR_START: TourState = { step: 0, done: false };
export const TOUR_DONE: TourState = { step: TOUR_STEPS.length - 1, done: true };

const forward = (state: TourState): TourState =>
	state.step >= TOUR_STEPS.length - 1 ? TOUR_DONE : { step: state.step + 1, done: false };

/**
 * Moves the tour along. Doing the current step's action completes it. Actions meant for
 * later steps are ignored, so trying things out of order never skips an explanation.
 */
export function tourReducer(state: TourState, event: TourEvent): TourState {
	if (event === 'restart') return TOUR_START;
	if (state.done) return state;
	switch (event) {
		case 'skip':
			return TOUR_DONE;
		case 'next':
			return forward(state);
		case 'back':
			return { step: Math.max(0, state.step - 1), done: false };
		default:
			return event === TOUR_STEPS[state.step].action ? forward(state) : state;
	}
}
