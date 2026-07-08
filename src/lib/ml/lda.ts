import type { Dataset } from '../../types/ml';

export interface LdaModel {
	means: [{ x: number; y: number }, { x: number; y: number }];
	/** Pooled within-class covariance as [xx, xy, yy]. */
	pooled: [number, number, number];
	/** Discriminant direction, proportional to the inverse covariance times the mean gap. */
	direction: [number, number];
	bias: number;
	/** Between-class separation relative to within-class spread along the direction. */
	fisherRatio: number;
}

/** Two-class LDA in closed form, assuming both classes share one covariance matrix. */
export function fitLda(data: Dataset): LdaModel {
	const groups = [0, 1].map((label) => data.filter((p) => p.label === label));
	const means = groups.map((g) => ({
		x: g.reduce((s, p) => s + p.x, 0) / Math.max(g.length, 1),
		y: g.reduce((s, p) => s + p.y, 0) / Math.max(g.length, 1),
	})) as LdaModel['means'];

	let [xx, xy, yy] = [1e-6, 0, 1e-6];
	groups.forEach((g, c) => {
		for (const p of g) {
			const dx = p.x - means[c].x;
			const dy = p.y - means[c].y;
			xx += (dx * dx) / data.length;
			xy += (dx * dy) / data.length;
			yy += (dy * dy) / data.length;
		}
	});

	const gapX = means[1].x - means[0].x;
	const gapY = means[1].y - means[0].y;
	const determinant = xx * yy - xy * xy;
	const direction: [number, number] = [(yy * gapX - xy * gapY) / determinant, (xx * gapY - xy * gapX) / determinant];

	const midpoint = direction[0] * (means[0].x + means[1].x) / 2 + direction[1] * (means[0].y + means[1].y) / 2;
	const priors = Math.log(Math.max(groups[1].length, 1) / Math.max(groups[0].length, 1));
	const between = (direction[0] * gapX + direction[1] * gapY) ** 2;
	const within = direction[0] ** 2 * xx + 2 * direction[0] * direction[1] * xy + direction[1] ** 2 * yy;
	return { means, pooled: [xx, xy, yy], direction, bias: priors - midpoint, fisherRatio: between / within };
}

/** Positive values favor class 1. */
export const ldaScore = (m: LdaModel, x: number, y: number): number => m.direction[0] * x + m.direction[1] * y + m.bias;
