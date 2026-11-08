import type { Dataset } from '../../../types/ml';
import type { AlgorithmSpec, Diagram, DiagramNode } from '../../../types/viz';
import { buildTree, classifyTree, gini, splitsAtLevel, treeDepth, type TreeNode } from '../../ml/decisionTree';
import { accuracy } from '../../ml/knn';
import { classPoints, fixed, percent } from '../scene';

interface State {
	depth: number;
	tree: TreeNode;
}

export function treeDiagram(root: TreeNode): Diagram {
	const nodes: DiagramNode[] = [];
	const edges: Diagram['edges'] = [];
	const levels = treeDepth(root) + 1;
	let leaves = 0;

	const visit = (node: TreeNode, level: number): number => {
		const index = nodes.length;
		nodes.push({ x: 0, y: levels === 1 ? 0.5 : level / (levels - 1), cls: node.prediction });
		if (node.kind === 'leaf') {
			nodes[index].x = leaves++;
			nodes[index].strength = 1;
		} else {
			nodes[index].label = `${node.feature} ≤ ${node.threshold.toFixed(2)}`;
			nodes[index].strength = 0.3;
			const left = visit(node.left, level + 1);
			const right = visit(node.right, level + 1);
			edges.push({ from: index, to: left }, { from: index, to: right });
			nodes[index].x = (nodes[left].x + nodes[right].x) / 2;
		}
		return index;
	};
	visit(root, 0);
	for (const node of nodes) node.x = leaves > 1 ? node.x / (leaves - 1) : 0.5;
	return { title: 'Tree structure', nodes, edges };
}

const classify = (s: State, data: Dataset) => accuracy(data, (x, y) => classifyTree(s.tree, x, y));

export const decisionTree: AlgorithmSpec<State> = {
	datasets: ['xor', 'moons', 'circles', 'linear', 'outliers'],
	params: [{ key: 'maxDepth', label: 'Max depth', value: 5, min: 1, max: 6, step: 1 }],
	labeled: true,
	holdout: true,
	speed: 1,
	thumbnailSteps: 4,
	legend: () => ['Class A', 'Class B'],
	init: (data) => ({ depth: 0, tree: buildTree(data, 0) }),
	step: (s, data) => ({ depth: s.depth + 1, tree: buildTree(data, s.depth + 1) }),
	isDone: (s, p) => s.depth >= p.maxDepth || treeDepth(s.tree) < s.depth,
	scene: (s, data) => ({ region: (x, y) => classifyTree(s.tree, x, y), points: classPoints(data) }),
	metrics: (s, data) => [
		{ label: 'Depth', value: String(treeDepth(s.tree)) },
		{ label: 'Training accuracy', value: percent(classify(s, data)) },
		{ label: 'Root Gini', value: s.tree.impurity.toFixed(3) },
	],
	caption(s, data) {
		if (s.depth === 0) return 'A single leaf predicts the majority class everywhere. Step to add a level of splits.';
		return `Depth ${treeDepth(s.tree)} fits ${percent(classify(s, data))} of the training points. Every split is axis-aligned.`;
	},
	diagram: (s) => treeDiagram(s.tree),
	calculations(s, data) {
		const columns = ['Node', 'n', 'Gini', 'Split', 'Left n · Gini', 'Right n · Gini', 'Weighted', 'Gain'];
		if (s.depth === 0) {
			return {
				title: 'The root node, before any split',
				columns: ['Samples', 'Gini'],
				rows: [[String(data.length), fixed(gini(data))]],
				note: 'Step to search every threshold on x and y for the split that lowers the Gini impurity the most.',
			};
		}
		const splits = splitsAtLevel(s.tree, data, s.depth);
		return {
			title: `Splits made at depth ${s.depth}`,
			columns,
			rows: splits.map((row) => [
				row.path.join(' → ') || 'Root',
				String(row.samples),
				fixed(row.gini),
				`${row.feature} ≤ ${row.threshold.toFixed(2)}`,
				`${row.left.samples} · ${fixed(row.left.gini)}`,
				`${row.right.samples} · ${fixed(row.right.gini)}`,
				fixed(row.weighted),
				fixed(row.gain),
			]),
			note: splits.length
				? 'n is the number of samples. Weighted = (n_left · Gini_left + n_right · Gini_right) / n, and Gain = Gini − Weighted. Each node keeps the threshold on x or y with the largest gain.'
				: 'No node could split further: every leaf is already pure, or too small to split.',
		};
	},
};
