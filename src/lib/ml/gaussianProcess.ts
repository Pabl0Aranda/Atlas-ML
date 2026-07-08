export interface GpParams {
	/** How far apart two inputs can be and still strongly influence each other. */
	lengthScale: number;
	/** Standard deviation of the observation noise. */
	noise: number;
}

export interface GpModel {
	xs: number[];
	/** Lower Cholesky factor of K + noise^2 I. */
	cholesky: number[][];
	/** (K + noise^2 I)^-1 y */
	alpha: number[];
	params: GpParams;
}

/** Squared exponential kernel with unit signal variance. */
export const rbf = (a: number, b: number, lengthScale: number): number =>
	Math.exp(-((a - b) ** 2) / (2 * lengthScale ** 2));

export function choleskyFactor(matrix: number[][]): number[][] {
	const n = matrix.length;
	const L = matrix.map((row) => row.map(() => 0));
	for (let i = 0; i < n; i++) {
		for (let j = 0; j <= i; j++) {
			let sum = matrix[i][j];
			for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
			L[i][j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / L[j][j];
		}
	}
	return L;
}

function solveLower(L: number[][], b: number[]): number[] {
	const x = [...b];
	for (let i = 0; i < x.length; i++) {
		for (let k = 0; k < i; k++) x[i] -= L[i][k] * x[k];
		x[i] /= L[i][i];
	}
	return x;
}

function solveUpperTransposed(L: number[][], b: number[]): number[] {
	const x = [...b];
	for (let i = x.length - 1; i >= 0; i--) {
		for (let k = i + 1; k < x.length; k++) x[i] -= L[k][i] * x[k];
		x[i] /= L[i][i];
	}
	return x;
}

export function fitGp(xs: number[], ys: number[], params: GpParams): GpModel {
	const K = xs.map((a, i) => xs.map((b, j) => rbf(a, b, params.lengthScale) + (i === j ? params.noise ** 2 : 0)));
	const cholesky = choleskyFactor(K);
	return { xs, cholesky, alpha: solveUpperTransposed(cholesky, solveLower(cholesky, ys)), params };
}

/** Posterior mean and variance of the function at x. */
export function gpPredict(model: GpModel, x: number): { mean: number; variance: number } {
	const k = model.xs.map((b) => rbf(x, b, model.params.lengthScale));
	const mean = k.reduce((s, v, i) => s + v * model.alpha[i], 0);
	const v = solveLower(model.cholesky, k);
	return { mean, variance: Math.max(1 - v.reduce((s, a) => s + a * a, 0), 0) };
}
