import type { Dataset, DataPoint } from '../../types/ml';

type Feature = 'x' | 'y';

const ALL_FEATURES: Feature[] = ['x', 'y'];

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
export function bestSplit(data: DataPoint[], features = ALL_FEATURES): { feature: Feature; threshold: number; score: number } | null {
	let best: { feature: Feature; threshold: number; score: number } | null = null;
	for (const feature of features) {
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

/** Recursively grows a CART tree down to maxDepth. Forests pass pickFeatures to limit each split. */
export function buildTree(data: Dataset, maxDepth: number, minSamples = 2, pickFeatures = () => ALL_FEATURES): TreeNode {
	const impurity = gini(data);
	const base = { prediction: majority(data), samples: data.length, impurity };
	const split = maxDepth > 0 && impurity > 0 && data.length >= minSamples ? bestSplit(data, pickFeatures()) : null;
	if (!split) return { kind: 'leaf', ...base };
	const { feature, threshold } = split;
	return {
		kind: 'split',
		...base,
		feature,
		threshold,
		left: buildTree(data.filter((p) => p[feature] <= threshold), maxDepth - 1, minSamples, pickFeatures),
		right: buildTree(data.filter((p) => p[feature] > threshold), maxDepth - 1, minSamples, pickFeatures),
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

export interface SplitStats {
	left: { samples: number; gini: number };
	right: { samples: number; gini: number };
	/** Each side's Gini weighted by its share of the samples. The tree picks the split that minimizes it. */
	weighted: number;
	/** How much the split lowers impurity: the parent's Gini minus the weighted Gini. */
	gain: number;
}

/** The arithmetic behind one split, as the calculations panel shows it. */
export function splitStats(data: DataPoint[], feature: Feature, threshold: number): SplitStats {
	const left = data.filter((p) => p[feature] <= threshold);
	const right = data.filter((p) => p[feature] > threshold);
	const weighted = data.length ? (left.length * gini(left) + right.length * gini(right)) / data.length : 0;
	return {
		left: { samples: left.length, gini: gini(left) },
		right: { samples: right.length, gini: gini(right) },
		weighted,
		gain: gini(data) - weighted,
	};
}

export interface LevelSplit extends SplitStats {
	/** The rules leading to this node from the root, such as "x ≤ 0.12". */
	path: string[];
	samples: number;
	gini: number;
	feature: Feature;
	threshold: number;
}

/** Every split made at one level of the tree, counting the root's split as level 1. */
export function splitsAtLevel(root: TreeNode, data: DataPoint[], level: number): LevelSplit[] {
	const found: LevelSplit[] = [];
	const visit = (node: TreeNode, rows: DataPoint[], depth: number, path: string[]) => {
		if (node.kind === 'leaf') return;
		const { feature, threshold } = node;
		if (depth === level) {
			found.push({ path, samples: rows.length, gini: gini(rows), feature, threshold, ...splitStats(rows, feature, threshold) });
			return;
		}
		const rule = threshold.toFixed(2);
		visit(node.left, rows.filter((p) => p[feature] <= threshold), depth + 1, [...path, `${feature} ≤ ${rule}`]);
		visit(node.right, rows.filter((p) => p[feature] > threshold), depth + 1, [...path, `${feature} > ${rule}`]);
	};
	if (level >= 1) visit(root, data, 1, []);
	return found;
}
