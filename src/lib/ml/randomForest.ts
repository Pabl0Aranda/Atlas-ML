import type { Dataset } from '../../types/ml';
import { buildTree, classifyTree, type TreeNode } from './decisionTree';
import { createRng } from './random';

const SEED = 41;
/** Spaces out tree seeds, so each tree can be regrown from its index alone. */
const SEED_STRIDE = 7919;

export interface ForestState {
	trees: TreeNode[];
	classes: number;
	/** Per sample, votes per class from the trees that never saw it. */
	oobVotes: number[][];
	oobAccuracy: number;
	oobHistory: number[];
}

/** Index of the largest count. Ties go to the lowest class. */
const winner = (votes: number[]) => votes.indexOf(Math.max(...votes));

export function initForest(data: Dataset): ForestState {
	const classes = Math.max(...data.map((p) => p.label), 1) + 1;
	return {
		trees: [],
		classes,
		oobVotes: data.map(() => new Array<number>(classes).fill(0)),
		oobAccuracy: 0,
		oobHistory: [],
	};
}

/** Accuracy over the samples that at least one tree left out of its bootstrap. */
function outOfBagAccuracy(votes: number[][], data: Dataset): number {
	const scored = data.filter((_, i) => votes[i].some((v) => v > 0));
	if (!scored.length) return 0;
	return data.filter((p, i) => votes[i].some((v) => v > 0) && winner(votes[i]) === p.label).length / scored.length;
}

/** Grows one tree on a bootstrap sample, choosing a random feature at every split. */
export function forestStep(state: ForestState, data: Dataset, maxDepth: number): ForestState {
	const rng = createRng(SEED + state.trees.length * SEED_STRIDE);
	const inBag = new Array<boolean>(data.length).fill(false);
	const sample = data.map(() => {
		const index = Math.floor(rng() * data.length);
		inBag[index] = true;
		return data[index];
	});
	const tree = buildTree(sample, maxDepth, 2, () => [rng() < 0.5 ? 'x' : 'y']);

	const oobVotes = state.oobVotes.map((votes, i) => {
		if (inBag[i]) return votes;
		const next = [...votes];
		next[classifyTree(tree, data[i].x, data[i].y)]++;
		return next;
	});
	const oobAccuracy = outOfBagAccuracy(oobVotes, data);
	return {
		...state,
		trees: [...state.trees, tree],
		oobVotes,
		oobAccuracy,
		oobHistory: [...state.oobHistory, oobAccuracy],
	};
}

/** Majority vote across the trees. */
export function forestClassify(state: ForestState, x: number, y: number): number {
	const votes = new Array<number>(state.classes).fill(0);
	for (const tree of state.trees) votes[classifyTree(tree, x, y)]++;
	return winner(votes);
}
