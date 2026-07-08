import type { Dataset } from '../../types/ml';
import { initKMeans } from './kmeans';

const MIN_VARIANCE = 1e-4;
const CONVERGENCE = 1e-5;

/** One 2D Gaussian with covariance [xx, xy, yy]. */
export interface Component {
	weight: number;
	x: number;
	y: number;
	cov: [number, number, number];
}

export interface GmmState {
	components: Component[];
	/** responsibilities[i][k]: the probability that component k produced point i. */
	responsibilities: number[][];
	iteration: number;
	logLikelihood: number;
	history: number[];
	converged: boolean;
}

export function density(c: Component, x: number, y: number): number {
	const [xx, xy, yy] = c.cov;
	const determinant = xx * yy - xy * xy;
	const dx = x - c.x;
	const dy = y - c.y;
	const mahalanobis = (yy * dx * dx - 2 * xy * dx * dy + xx * dy * dy) / determinant;
	return Math.exp(-0.5 * mahalanobis) / (2 * Math.PI * Math.sqrt(determinant));
}

/** E-step: soft-assign each point, and measure how well the mixture explains the data. */
function expectation(data: Dataset, components: Component[]) {
	let logLikelihood = 0;
	const responsibilities = data.map((p) => {
		const parts = components.map((c) => c.weight * density(c, p.x, p.y));
		const total = parts.reduce((a, b) => a + b, 0) || 1e-300;
		logLikelihood += Math.log(total);
		return parts.map((v) => v / total);
	});
	return { responsibilities, logLikelihood };
}

export function initGmm(data: Dataset, k: number, seed = 11): GmmState {
	const components = initKMeans(data, k, seed).centroids.map(
		(c): Component => ({ weight: 1 / k, x: c.x, y: c.y, cov: [0.1, 0, 0.1] }),
	);
	const { responsibilities, logLikelihood } = expectation(data, components);
	return { components, responsibilities, iteration: 0, logLikelihood, history: [logLikelihood], converged: false };
}

/** One EM iteration: refit every component from its responsibilities, then reassign. */
export function gmmStep(s: GmmState, data: Dataset): GmmState {
	const components = s.components.map((_, k): Component => {
		const weightOf = (i: number) => s.responsibilities[i][k];
		const total = data.reduce((sum, _p, i) => sum + weightOf(i), 0) || 1e-12;
		const x = data.reduce((sum, p, i) => sum + weightOf(i) * p.x, 0) / total;
		const y = data.reduce((sum, p, i) => sum + weightOf(i) * p.y, 0) / total;
		let [xx, xy, yy] = [0, 0, 0];
		data.forEach((p, i) => {
			xx += weightOf(i) * (p.x - x) ** 2;
			xy += weightOf(i) * (p.x - x) * (p.y - y);
			yy += weightOf(i) * (p.y - y) ** 2;
		});
		return {
			weight: total / data.length,
			x,
			y,
			cov: [xx / total + MIN_VARIANCE, xy / total, yy / total + MIN_VARIANCE],
		};
	});
	const { responsibilities, logLikelihood } = expectation(data, components);
	return {
		components,
		responsibilities,
		iteration: s.iteration + 1,
		logLikelihood,
		history: [...s.history, logLikelihood],
		converged: Math.abs(logLikelihood - s.logLikelihood) < CONVERGENCE,
	};
}

export const mostLikely = (row: number[]): number => row.indexOf(Math.max(...row));

/** Traces a contour of constant Mahalanobis distance, via the Cholesky factor of the covariance. */
export function ellipse(c: Component, radius = 2, points = 48): [number, number][] {
	const [xx, xy, yy] = c.cov;
	const a = Math.sqrt(xx);
	const b = xy / a;
	const d = Math.sqrt(Math.max(yy - b * b, 0));
	return Array.from({ length: points + 1 }, (_, i) => {
		const t = (i / points) * 2 * Math.PI;
		const u = radius * Math.cos(t);
		const v = radius * Math.sin(t);
		return [c.x + a * u, c.y + b * u + d * v];
	});
}
