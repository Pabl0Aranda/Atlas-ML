import type { Dataset } from '../../types/ml';

const MIN_VARIANCE = 1e-3;

export type Feature = 'x' | 'y';

export interface ClassStats {
	label: number;
	prior: number;
	mean: Record<Feature, number>;
	variance: Record<Feature, number>;
}

/** Estimates a prior plus an independent Gaussian per feature, for each class. */
export function fitNaiveBayes(data: Dataset): ClassStats[] {
	const labels = [...new Set(data.map((p) => p.label))].sort((a, b) => a - b);
	return labels.map((label) => {
		const members = data.filter((p) => p.label === label);
		const stats = (feature: Feature) => {
			const mean = members.reduce((s, p) => s + p[feature], 0) / members.length;
			const variance = members.reduce((s, p) => s + (p[feature] - mean) ** 2, 0) / members.length;
			return { mean, variance: Math.max(variance, MIN_VARIANCE) };
		};
		const x = stats('x');
		const y = stats('y');
		return {
			label,
			prior: members.length / data.length,
			mean: { x: x.mean, y: y.mean },
			variance: { x: x.variance, y: y.variance },
		};
	});
}

function logGaussian(value: number, mean: number, variance: number): number {
	return -0.5 * (Math.log(2 * Math.PI * variance) + (value - mean) ** 2 / variance);
}

/** log P(class) + sum of log P(feature | class), using only the listed features. */
export function logPosterior(c: ClassStats, x: number, y: number, features: Feature[]): number {
	const point = { x, y };
	return features.reduce((s, f) => s + logGaussian(point[f], c.mean[f], c.variance[f]), Math.log(c.prior));
}

export function classifyNaiveBayes(model: ClassStats[], x: number, y: number, features: Feature[] = ['x', 'y']): number {
	let best = model[0];
	for (const c of model) if (logPosterior(c, x, y, features) > logPosterior(best, x, y, features)) best = c;
	return best.label;
}
