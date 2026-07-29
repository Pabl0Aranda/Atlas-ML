import type { Dataset } from '../../types/ml';

/** Floor on the weighted error, so a perfect stump gets a large but finite vote. */
const MIN_ERROR = 1e-10;

export interface Stump {
	feature: 'x' | 'y';
	threshold: number;
	/** +1 predicts the positive class above the threshold, -1 below it. */
	polarity: 1 | -1;
	/** This stump's say in the final vote. */
	alpha: number;
}

export interface AdaBoostState {
	stumps: Stump[];
	/** Sample weights, always summing to 1. */
	weights: number[];
	/** Weighted error of the latest stump. */
	error: number;
	/** Share of training samples the ensemble misclassified, per round. */
	errorHistory: number[];
	/** True once a stump is perfect, or none beats chance. */
	finished: boolean;
}

const sign = (label: number) => (label > 0 ? 1 : -1);

const stumpVote = (s: Stump, x: number, y: number) => ((s.feature === 'x' ? x : y) > s.threshold ? s.polarity : -s.polarity);

export function initAdaBoost(data: Dataset): AdaBoostState {
	return { stumps: [], weights: data.map(() => 1 / data.length), error: 0.5, errorHistory: [], finished: false };
}

/** Sweeps every threshold on both features for the lowest weighted error. */
function bestStump(data: Dataset, weights: number[]): Omit<Stump, 'alpha'> & { error: number } {
	let best: Omit<Stump, 'alpha'> & { error: number } = { feature: 'x', threshold: 0, polarity: 1, error: Infinity };
	for (const feature of ['x', 'y'] as const) {
		const order = data.map((_, i) => i).sort((a, b) => data[a][feature] - data[b][feature]);
		// With polarity +1 and every sample above the threshold, the negatives are the mistakes.
		let error = order.reduce((sum, i) => sum + (sign(data[i].label) < 0 ? weights[i] : 0), 0);
		const consider = (threshold: number) => {
			if (error < best.error) best = { feature, threshold, polarity: 1, error };
			if (1 - error < best.error) best = { feature, threshold, polarity: -1, error: 1 - error };
		};
		consider(data[order[0]][feature] - 1);
		for (let n = 0; n < order.length - 1; n++) {
			const i = order[n];
			error += sign(data[i].label) * weights[i];
			const [low, high] = [data[i][feature], data[order[n + 1]][feature]];
			if (low !== high) consider((low + high) / 2);
		}
	}
	return best;
}

/** One boosting round: fit a stump, give it a vote, then reweight the samples. */
export function adaBoostStep(state: AdaBoostState, data: Dataset): AdaBoostState {
	if (state.finished) return state;
	const { error, ...stump } = bestStump(data, state.weights);
	if (error >= 0.5 - 1e-12) return { ...state, error, finished: true };

	const clipped = Math.max(error, MIN_ERROR);
	const added: Stump = { ...stump, alpha: 0.5 * Math.log((1 - clipped) / clipped) };
	const raised = state.weights.map(
		(w, i) => w * Math.exp(-added.alpha * sign(data[i].label) * stumpVote(added, data[i].x, data[i].y)),
	);
	const total = raised.reduce((sum, w) => sum + w, 0);

	const next: AdaBoostState = {
		...state,
		stumps: [...state.stumps, added],
		weights: raised.map((w) => w / total),
		error,
		finished: error <= MIN_ERROR,
	};
	const wrong = data.filter((p) => adaBoostClassify(next, p.x, p.y) !== (p.label > 0 ? 1 : 0)).length;
	return { ...next, errorHistory: [...state.errorHistory, wrong / data.length] };
}

/** Sign of the alpha-weighted vote of all stumps. */
export function adaBoostClassify(state: Pick<AdaBoostState, 'stumps'>, x: number, y: number): number {
	return state.stumps.reduce((sum, s) => sum + s.alpha * stumpVote(s, x, y), 0) > 0 ? 1 : 0;
}
