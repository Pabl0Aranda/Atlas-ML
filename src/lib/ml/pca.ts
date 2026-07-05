import type { Dataset } from '../../types/ml';
import { mean, norm } from './linalg';

export interface PcaState {
	center: { x: number; y: number };
	/** 2x2 sample covariance as [[xx, xy], [xy, yy]]. */
	covariance: [[number, number], [number, number]];
	/** Current estimate of the first principal axis (unit length). */
	axis: [number, number];
	iteration: number;
	/** Share of total variance captured along the current axis. */
	explained: number;
	explainedHistory: number[];
}

export function covariance2d(data: Dataset) {
	const cx = mean(data.map((p) => p.x));
	const cy = mean(data.map((p) => p.y));
	const n = Math.max(data.length - 1, 1);
	let xx = 0;
	let xy = 0;
	let yy = 0;
	for (const p of data) {
		xx += (p.x - cx) ** 2;
		xy += (p.x - cx) * (p.y - cy);
		yy += (p.y - cy) ** 2;
	}
	return { center: { x: cx, y: cy }, covariance: [[xx / n, xy / n], [xy / n, yy / n]] as PcaState['covariance'] };
}

function explainedBy(axis: [number, number], c: PcaState['covariance']): number {
	const [u, v] = axis;
	const variance = u * u * c[0][0] + 2 * u * v * c[0][1] + v * v * c[1][1];
	return variance / Math.max(c[0][0] + c[1][1], 1e-12);
}

export function initPca(data: Dataset, startAngle = 2.4): PcaState {
	const { center, covariance } = covariance2d(data);
	const axis: [number, number] = [Math.cos(startAngle), Math.sin(startAngle)];
	const explained = explainedBy(axis, covariance);
	return { center, covariance, axis, iteration: 0, explained, explainedHistory: [explained] };
}

/** One power iteration: multiplying by the covariance rotates the axis toward PC1. */
export function pcaStep(state: PcaState): PcaState {
	const c = state.covariance;
	const [u, v] = state.axis;
	const raw = [c[0][0] * u + c[0][1] * v, c[0][1] * u + c[1][1] * v];
	const length = norm(raw) || 1;
	const axis: [number, number] = [raw[0] / length, raw[1] / length];
	const explained = explainedBy(axis, c);
	return {
		...state,
		axis,
		iteration: state.iteration + 1,
		explained,
		explainedHistory: [...state.explainedHistory, explained],
	};
}

/** Closed-form eigenvalues of a symmetric 2x2 matrix, largest first. */
export function eigenvalues2d(c: PcaState['covariance']): [number, number] {
	const half = (c[0][0] + c[1][1]) / 2;
	const root = Math.sqrt(((c[0][0] - c[1][1]) / 2) ** 2 + c[0][1] ** 2);
	return [half + root, half - root];
}

/** Orthogonal projection of a point onto the line through center along axis. */
export function project(state: PcaState, x: number, y: number): { x: number; y: number } {
	const [u, v] = state.axis;
	const t = (x - state.center.x) * u + (y - state.center.y) * v;
	return { x: state.center.x + t * u, y: state.center.y + t * v };
}
