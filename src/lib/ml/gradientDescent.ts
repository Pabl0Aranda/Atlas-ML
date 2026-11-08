/** A loss over two weights, with its gradient, drawn on the plot's square from -1 to 1. */
export interface Surface {
	loss(x: number, y: number): number;
	gradient(x: number, y: number): [number, number];
}

/** The landscapes the sandbox offers, each showing a different way descent behaves. */
export const SURFACES: Surface[] = [
	// A round bowl: the gradient points straight at the minimum from everywhere.
	{ loss: (x, y) => x * x + y * y, gradient: (x, y) => [2 * x, 2 * y] },
	// A narrow valley: steep across, shallow along, so large steps zigzag.
	{ loss: (x, y) => 0.5 * x * x + 4.5 * y * y, gradient: (x, y) => [x, 9 * y] },
	// Two valleys, the left one deeper: where you start decides where you end.
	{
		loss: (x, y) => (x * x - 0.5) ** 2 + 0.15 * x + 0.5 * y * y + 0.15,
		gradient: (x, y) => [4 * x * (x * x - 0.5) + 0.15, y],
	},
	// A curved valley, a scaled Rosenbrock function, whose floor bends away from the gradient.
	{
		loss: (x, y) => (0.5 - x) ** 2 + 10 * (y - x * x) ** 2,
		gradient: (x, y) => [-2 * (0.5 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
	},
];

export interface DescentState {
	x: number;
	y: number;
	/** Velocity, for momentum. It stays zero without momentum. */
	vx: number;
	vy: number;
	step: number;
	/** Every position visited, the start included. */
	path: [number, number][];
	/** The loss at each position in the path. */
	losses: number[];
	diverged: boolean;
}

/** Beyond this distance from the center, the run has clearly blown up. */
const DIVERGED_AT = 4;

export function startDescent(surface: Surface, x: number, y: number): DescentState {
	return { x, y, vx: 0, vy: 0, step: 0, path: [[x, y]], losses: [surface.loss(x, y)], diverged: false };
}

/**
 * One update with momentum β: the velocity keeps a share β of the last step, then adds a
 * step of size α against the gradient. With β = 0 this is plain gradient descent.
 */
export function descentStep(state: DescentState, surface: Surface, alpha: number, beta: number): DescentState {
	if (state.diverged) return state;
	const [gx, gy] = surface.gradient(state.x, state.y);
	const vx = beta * state.vx - alpha * gx;
	const vy = beta * state.vy - alpha * gy;
	const x = state.x + vx;
	const y = state.y + vy;
	const loss = surface.loss(x, y);
	if (!Number.isFinite(loss) || Math.hypot(x, y) > DIVERGED_AT) return { ...state, step: state.step + 1, diverged: true };
	return { x, y, vx, vy, step: state.step + 1, path: [...state.path, [x, y]], losses: [...state.losses, loss], diverged: false };
}

export const gradientSize = (surface: Surface, x: number, y: number): number => Math.hypot(...surface.gradient(x, y));

/** The lowest loss on a fine grid over the plot, so contour levels can start at the floor. */
export function lowestLoss(surface: Surface, resolution = 120): number {
	let lowest = Infinity;
	for (let r = 0; r <= resolution; r++) {
		for (let c = 0; c <= resolution; c++) lowest = Math.min(lowest, surface.loss(-1 + (2 * c) / resolution, -1 + (2 * r) / resolution));
	}
	return lowest;
}
