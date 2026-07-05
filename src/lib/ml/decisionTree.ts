import type { Dataset, DataPoint } from '../../types/ml';

export type TreeNode =
	| { kind: 'leaf'; prediction: number; samples: number; impurity: number }
	| {
			kind: 'split';
			feature: 'x' | 'y';
			threshold: number;
			prediction: number;
			samples: number;
			impurity: number;
			left: TreeNode;
			right: TreeNode;
	  };

function counts(data: DataPoint[]): Map<number, number> {
	const result = new Map<number, number>();
	for (const p of data) result.set(p.label, (result.get(p.label) ?? 0) + 1);
	return result;
}

/** Gini impurity: the chance two random samples from the node disagree. */
export function gini(data: DataPoint[]): number {
	if (!data.length) return 0;
	let sum = 0;
	for (const c of counts(data).values()) sum += (c / data.length) ** 2;
	return 1 - sum;
}

function majority(data: DataPoint[]): number {
	let best = 0;
	let bestCount = -1;
	for (const [label, c] of counts(data)) {
		if (c > bestCount || (c === bestCount && label < best)) {
			best = label;
			bestCount = c;
		}
	}
	return best;
}

/** Finds the axis-aligned split with the lowest weighted child impurity. */
export function bestSplit(data: DataPoint[]): { feature: 'x' | 'y'; threshold: number; score: number } | null {
	let best: { feature: 'x' | 'y'; threshold: number; score: number } | null = null;
	for (const feature of ['x', 'y'] as const) {
		const sorted = [...data].sort((a, b) => a[feature] - b[feature]);
		for (let i = 1; i < sorted.length; i++) {
			if (sorted[i][feature] === sorted[i - 1][feature]) continue;
			const left = sorted.slice(0, i);
			const right = sorted.slice(i);
			const score = (left.length * gini(left) + right.length * gini(right)) / sorted.length;
			if (!best || score < best.score) {
				best = { feature, threshold: (sorted[i][feature] + sorted[i - 1][feature]) / 2, score };
			}
		}
	}
	return best;
}

/** Recursively grows a CART tree down to maxDepth. */
export function buildTree(data: Dataset, maxDepth: number, minSamples = 2): TreeNode {
	const impurity = gini(data);
	const base = { prediction: majority(data), samples: data.length, impurity };
	const split = maxDepth > 0 && impurity > 0 && data.length >= minSamples ? bestSplit(data) : null;
	if (!split) return { kind: 'leaf', ...base };
	const { feature, threshold } = split;
	return {
		kind: 'split',
		...base,
		feature,
		threshold,
		left: buildTree(data.filter((p) => p[feature] <= threshold), maxDepth - 1, minSamples),
		right: buildTree(data.filter((p) => p[feature] > threshold), maxDepth - 1, minSamples),
	};
}

export function classifyTree(node: TreeNode, x: number, y: number): number {
	let current = node;
	while (current.kind === 'split') {
		current = (current.feature === 'x' ? x : y) <= current.threshold ? current.left : current.right;
	}
	return current.prediction;
}

export function treeDepth(node: TreeNode): number {
	return node.kind === 'leaf' ? 0 : 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
}
