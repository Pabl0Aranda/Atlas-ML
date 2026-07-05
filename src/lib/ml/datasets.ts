import type { Dataset, DatasetKind } from '../../types/ml';
import { createRng, gaussian } from './random';

export const DATASET_LABELS: Record<DatasetKind, string> = {
	linear: 'Linear',
	moons: 'Moons',
	circles: 'Circles',
	xor: 'XOR',
	outliers: 'Noise / outliers',
	trend: 'Linear trend',
	curve: 'Curve',
	'trend-outliers': 'Trend + outliers',
	blobs: 'Blobs',
	ellipse: 'Correlated cloud',
	dirty: 'Cloud with outliers',
	wide: 'Mismatched scales',
	imbalanced: 'Imbalanced classes',
};

const clamp = (v: number) => Math.max(-0.98, Math.min(0.98, v));

/** Generates a deterministic dataset in the [-1, 1] domain. */
export function generateDataset(kind: DatasetKind, n = 120, seed = 7): Dataset {
	const rng = createRng(seed);
	const noise = (s: number) => gaussian(rng) * s;
	const point = (x: number, y: number, label = 0) => ({ x: clamp(x), y: clamp(y), label });
	const half = (i: number) => (i < n / 2 ? 0 : 1);
	const each = (fn: (i: number) => Dataset[number]) => Array.from({ length: n }, (_, i) => fn(i));

	switch (kind) {
		case 'linear':
			return each((i) => {
				const c = half(i) ? 0.42 : -0.42;
				return point(c + noise(0.2), c + noise(0.2), half(i));
			});
		case 'outliers':
			return each((i) => {
				const c = half(i) ? 0.42 : -0.42;
				const flipped = rng() < 0.1;
				return point(c + noise(0.24), c + noise(0.24), flipped ? 1 - half(i) : half(i));
			});
		case 'moons':
			return each((i) => {
				const t = rng() * Math.PI;
				const [x, y] = half(i) ? [1 - Math.cos(t), 0.5 - Math.sin(t)] : [Math.cos(t), Math.sin(t)];
				return point(0.6 * x - 0.3 + noise(0.05), 0.6 * y - 0.15 + noise(0.05), half(i));
			});
		case 'circles':
			return each((i) => {
				const t = rng() * 2 * Math.PI;
				const r = (half(i) ? 0.75 : 0.3) + noise(0.05);
				return point(r * Math.cos(t), r * Math.sin(t), half(i));
			});
		case 'xor':
			return each(() => {
				const x = rng() * 1.8 - 0.9;
				const y = rng() * 1.8 - 0.9;
				return point(x, y, x * y > 0 ? 0 : 1);
			});
		case 'blobs': {
			const centers = [[-0.5, 0.45], [0.5, 0.5], [0.05, -0.5], [-0.6, -0.4]];
			return each((i) => {
				const [cx, cy] = centers[i % centers.length];
				return point(cx + noise(0.13), cy + noise(0.13));
			});
		}
		case 'ellipse':
			return each(() => {
				const major = noise(0.42);
				const minor = noise(0.11);
				const angle = 0.6;
				return point(
					major * Math.cos(angle) - minor * Math.sin(angle),
					major * Math.sin(angle) + minor * Math.cos(angle),
				);
			});
		case 'dirty':
			return each((i) => {
				if (i % 15 === 0) {
					const t = rng() * 2 * Math.PI;
					return point(0.85 * Math.cos(t), 0.85 * Math.sin(t));
				}
				return point(noise(0.16), noise(0.16));
			});
		case 'wide':
			return each(() => point(noise(0.4), 0.55 + noise(0.05)));
		case 'imbalanced':
			return each((i) => {
				const rare = i % 10 === 0;
				return point((rare ? 0.45 : -0.15) + noise(0.24), (rare ? 0.45 : -0.15) + noise(0.24), rare ? 1 : 0);
			});
		case 'trend':
		case 'trend-outliers':
			return each(() => {
				const x = rng() * 1.8 - 0.9;
				const outlier = kind === 'trend-outliers' && rng() < 0.08;
				return point(x, 0.6 * x - 0.1 + noise(0.08) + (outlier ? 0.9 : 0));
			});
		case 'curve':
			return each(() => {
				const x = rng() * 1.8 - 0.9;
				return point(x, 0.7 * Math.sin(3 * x) + noise(0.07));
			});
	}
}
