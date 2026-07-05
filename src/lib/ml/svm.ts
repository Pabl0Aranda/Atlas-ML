import type { Dataset } from '../../types/ml';
import { createRng } from './random';

export type Kernel = 'linear' | 'rbf';

export interface SvmState {
	/** Dual variables, one per sample. Non-zero entries are support vectors. */
	alphas: number[];
	bias: number;
	pass: number;
	/** How many alpha pairs the last pass changed. Zero means converged. */
	changed: number;
	rng: () => number;
}

export interface SvmParams {
	C: number;
	kernel: Kernel;
	gamma: number;
}

const sign = (label: number) => (label === 1 ? 1 : -1);

function kernelValue(params: SvmParams, ax: number, ay: number, bx: number, by: number): number {
	if (params.kernel === 'linear') return ax * bx + ay * by;
	return Math.exp(-params.gamma * ((ax - bx) ** 2 + (ay - by) ** 2));
}

/** Signed decision value f(x). The margins sit at f = +1 and f = -1. */
export function svmDecision(state: SvmState, data: Dataset, params: SvmParams, x: number, y: number): number {
	let sum = state.bias;
	data.forEach((p, i) => {
		if (state.alphas[i] > 0) sum += state.alphas[i] * sign(p.label) * kernelValue(params, p.x, p.y, x, y);
	});
	return sum;
}

export function initSvm(data: Dataset, seed = 3): SvmState {
	return { alphas: data.map(() => 0), bias: 0, pass: 0, changed: -1, rng: createRng(seed) };
}

/** One sweep of simplified SMO: optimizes pairs of dual variables analytically. */
export function svmStep(state: SvmState, data: Dataset, params: SvmParams, tolerance = 1e-3): SvmState {
	const next: SvmState = { ...state, alphas: [...state.alphas], pass: state.pass + 1, changed: 0 };
	const { C } = params;
	const K = (i: number, j: number) => kernelValue(params, data[i].x, data[i].y, data[j].x, data[j].y);
	const error = (i: number) => svmDecision(next, data, params, data[i].x, data[i].y) - sign(data[i].label);

	for (let i = 0; i < data.length; i++) {
		const yi = sign(data[i].label);
		const Ei = error(i);
		const violatesKkt =
			(yi * Ei < -tolerance && next.alphas[i] < C) || (yi * Ei > tolerance && next.alphas[i] > 0);
		if (!violatesKkt) continue;

		let j = Math.floor(next.rng() * (data.length - 1));
		if (j >= i) j += 1;
		const yj = sign(data[j].label);
		const Ej = error(j);
		const ai = next.alphas[i];
		const aj = next.alphas[j];

		const low = yi === yj ? Math.max(0, ai + aj - C) : Math.max(0, aj - ai);
		const high = yi === yj ? Math.min(C, ai + aj) : Math.min(C, C + aj - ai);
		if (low >= high) continue;
		const eta = 2 * K(i, j) - K(i, i) - K(j, j);
		if (eta >= 0) continue;

		const newAj = Math.min(high, Math.max(low, aj - (yj * (Ei - Ej)) / eta));
		if (Math.abs(newAj - aj) < 1e-5) continue;
		const newAi = ai + yi * yj * (aj - newAj);
		next.alphas[i] = newAi;
		next.alphas[j] = newAj;

		const b1 = next.bias - Ei - yi * (newAi - ai) * K(i, i) - yj * (newAj - aj) * K(i, j);
		const b2 = next.bias - Ej - yi * (newAi - ai) * K(i, j) - yj * (newAj - aj) * K(j, j);
		if (newAi > 0 && newAi < C) next.bias = b1;
		else if (newAj > 0 && newAj < C) next.bias = b2;
		else next.bias = (b1 + b2) / 2;
		next.changed += 1;
	}
	return next;
}

export const supportVectorCount = (state: SvmState): number => state.alphas.filter((a) => a > 1e-6).length;
