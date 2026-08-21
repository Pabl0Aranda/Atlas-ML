import type { DatasetKind, Hyperparameters } from '../../types/ml';
import type { AlgorithmId } from '../../types/viz';
import type { ComparePreset } from './compareSetup';

/** A guided starting point that makes one idea visible. */
export interface Preset {
	label: string;
	/** What to do and what to look for. Figures come from running this preset on the default seed. */
	note: string;
	dataset?: DatasetKind;
	params?: Hyperparameters;
	holdout?: boolean;
}

export const PRESETS: Partial<Record<AlgorithmId, Preset[]>> = {
	'linear-regression': [
		{ label: 'Learning rate too high', note: 'Press Play. Each update overshoots further than the last, and the loss diverges instead of falling.', params: { learningRate: 1.2 } },
		{ label: 'Learning rate too low', note: 'Press Play. The line does head the right way, but it creeps. Watch how long the loss curve takes to flatten.', params: { learningRate: 0.01 } },
	],
	'polynomial-regression': [
		{ label: 'Underfit', note: 'A degree 1 model is a straight line, so no amount of training can follow the curve.', dataset: 'curve', params: { degree: 1 } },
		{ label: 'High degree', note: 'Degree 12 has the freedom to chase the noise, and it reaches about a tenth of the underfit model’s loss on the training data.', dataset: 'curve', params: { degree: 12, l2: 0 } },
		{ label: 'Heavy regularizer', note: 'The same degree 12 model with a strong L2 penalty. Its weights stay small, which flattens the curve.', dataset: 'curve', params: { degree: 12, l2: 0.05 } },
	],
	knn: [
		{ label: 'Noisy labels', note: 'Press Play and watch Test accuracy. It starts near 83% at k = 1, where each noisy sample claims its own patch, and reaches about 97% by k = 9.', dataset: 'outliers', holdout: true },
		{ label: 'Too smooth', note: 'Press Play. Test accuracy starts at 100% on these clean moons, then falls to about 87% as a large k blurs the two curves together.', dataset: 'moons', holdout: true },
	],
	'decision-tree': [
		{ label: 'Overfit on noise', note: 'Press Play. By depth 6 the tree fits every training sample, yet Test accuracy is only about 90%.', dataset: 'outliers', params: { maxDepth: 6 }, holdout: true },
		{ label: 'Greedy on XOR', note: 'Press Step. One split reaches only about 62%, because neither feature alone explains XOR. Raise Max depth to see why greedy trees still get there.', dataset: 'xor', params: { maxDepth: 1 } },
	],
	'random-forest': [
		{ label: 'A single tree', note: 'Press Step once. One tree on one bootstrap sample scores about 87% on held-out data, and its boundary is jagged.', dataset: 'outliers', params: { trees: 1, maxDepth: 8 }, holdout: true },
		{ label: 'A big forest', note: 'Press Play. Training accuracy hits 100%, which says little. Out-of-bag accuracy settles near 94%, and Test accuracy near 90%.', dataset: 'outliers', params: { trees: 60, maxDepth: 8 }, holdout: true },
	],
	'gradient-boosting': [
		{ label: 'Fast and greedy', note: 'Press Play. With a learning rate of 1 and deeper trees, the training loss collapses to almost zero. Test accuracy reaches 90% and goes no further.', dataset: 'outliers', params: { rounds: 150, learningRate: 1, maxDepth: 4 }, holdout: true },
		{ label: 'Slow and steady', note: 'Press Play. Small steps climb from about 80% to the same 90% Test accuracy, using far gentler corrections.', dataset: 'outliers', params: { rounds: 150, learningRate: 0.05, maxDepth: 2 }, holdout: true },
	],
	adaboost: [
		{ label: 'Memorizing XOR', note: 'Press Play. Training accuracy climbs to about 77%, but Test accuracy stays near 50%. Stumps cannot express XOR, so that gain is memorized noise.', dataset: 'xor', holdout: true },
		{ label: 'Hard samples', note: 'Press Play and watch the points. Samples the stumps keep missing swell in size, and they cluster along the true boundary.', dataset: 'moons' },
	],
	svm: [
		{ label: 'Linear on moons', note: 'Press Play. A straight boundary cannot follow the curves, and it ends at about 83% on held-out data.', dataset: 'moons', params: { kernel: 0 }, holdout: true },
		{ label: 'RBF on moons', note: 'Press Play. The RBF kernel bends the boundary around each moon, and gets every held-out sample right.', dataset: 'moons', params: { kernel: 1, gamma: 5, C: 1 }, holdout: true },
		{ label: 'Overfit RBF', note: 'Press Play. A narrow kernel and high penalty wrap tight islands around noisy samples: about 98% on training data and 87% on held-out data.', dataset: 'outliers', params: { kernel: 1, gamma: 30, C: 20 }, holdout: true },
	],
	perceptron: [
		{ label: 'Separable data', note: 'Press Play. A line exists, so the perceptron is guaranteed to find one, and here it takes only 3 epochs.', dataset: 'linear' },
		{ label: 'No line exists', note: 'Press Play. On XOR the boundary keeps swinging, and accuracy jumps around without ever settling. This failure helped motivate multi-layer networks.', dataset: 'xor' },
	],
	mlp: [
		{ label: 'Too few units', note: 'Press Play. Two hidden units cannot wrap around a circle: after 3,000 epochs Test accuracy is only about 73%.', dataset: 'circles', params: { layers: 1, units: 2 }, holdout: true },
		{ label: 'Enough units', note: 'Press Play. Two layers of six units eventually enclose the inner circle, and reach 100% Test accuracy.', dataset: 'circles', params: { layers: 2, units: 6 }, holdout: true },
	],
	'logistic-regression': [
		{ label: 'Straight line on circles', note: 'Press Play. Linear features cannot separate rings, so accuracy stays near chance.', dataset: 'circles', params: { degree: 1 }, holdout: true },
		{ label: 'Quadratic features', note: 'Press Play. Squared features turn the boundary into a curve, and the same model reaches 100% on held-out data.', dataset: 'circles', params: { degree: 2 }, holdout: true },
	],
	'k-means': [
		{ label: 'Fails on moons', note: 'Press Play. Cluster boundaries are always straight, so k-means slices across both moons. Compare with DBSCAN on the same data.', dataset: 'moons', params: { k: 2 } },
		{ label: 'Round clusters', note: 'Press Play. Compact, similar-sized blobs are what k-means assumes, so it settles within a few iterations.', dataset: 'blobs', params: { k: 4 } },
	],
	dbscan: [
		{ label: 'ε too small', note: 'Press Play. Each moon shatters: 6 clusters, and 51 samples written off as noise.', dataset: 'moons', params: { eps: 0.08 } },
		{ label: 'ε too large', note: 'Press Play. Neighborhoods bridge the gap, so both moons merge into 1 cluster.', dataset: 'moons', params: { eps: 0.4 } },
		{ label: 'Uneven density', note: 'Press Play. One radius has to suit both rings. The sparser outer ring breaks up, for 8 clusters in total.', dataset: 'circles', params: { eps: 0.15 } },
	],
	'hierarchical-clustering': [
		{ label: 'Follow the moons', note: 'Press Play. Single linkage needs just one close pair to merge, so it traces each moon and recovers both exactly.', dataset: 'moons', params: { linkage: 0 } },
		{ label: 'Compact clusters', note: 'Press Play. Complete linkage penalizes long shapes, so it cuts across the moons. Only about 85% of samples land with their own moon.', dataset: 'moons', params: { linkage: 1 } },
	],
	'ridge-regression': [
		{ label: 'No penalty', note: 'Press Play. With λ = 0 nothing restrains the weights, and the largest grows to about 2.6.', params: { lambda: 0 } },
		{ label: 'Heavy penalty', note: 'Press Play. The largest weight stays near 0.5, yet all 9 terms remain non-zero. Ridge shrinks weights without removing them.', params: { lambda: 0.2 } },
	],
	'lasso-regression': [
		{ label: 'Heavy penalty', note: 'Press Play. The L1 penalty drives weights to exactly zero, leaving only 1 of 9 terms. That is the difference from ridge.', params: { lambda: 0.2 } },
	],
};

export const COMPARE_PRESETS: ComparePreset[] = [
	{ label: 'KNN vs SVM', note: 'Press Play. KNN raises k as it runs, so its Test accuracy starts at 100% and falls to about 87% by k = 41. The RBF SVM finishes at 100%.', a: 'knn', b: 'svm', dataset: 'moons', paramsB: { kernel: 1 }, holdout: true },
	{ label: 'Ridge vs Lasso', note: 'Press Play. Both use the same penalty, λ = 0.05. Ridge shrinks weights but keeps all 9 terms, while Lasso keeps only 1.', a: 'ridge-regression', b: 'lasso-regression', dataset: 'curve', paramsA: { lambda: 0.05 }, paramsB: { lambda: 0.05 } },
	{ label: 'k-means vs DBSCAN', note: 'Press Play. k-means can only draw straight boundaries, so about 83% of samples land with their own moon. DBSCAN follows the curves and gets about 99%, leaving 1 point as noise.', a: 'k-means', b: 'dbscan', dataset: 'moons', paramsA: { k: 2 } },
	{ label: 'Tree vs Forest', note: 'Press Play. Both end at 90% Test accuracy here, and the forest touches about 97% along the way. With only 30 held-out samples, one split cannot settle which is better.', a: 'decision-tree', b: 'random-forest', dataset: 'outliers', holdout: true },
	{ label: 'Linear vs RBF SVM', note: 'Press Play. The same algorithm with a different kernel: the straight boundary ends at about 83% Test accuracy, and the curved one at 100%.', a: 'svm', b: 'svm', dataset: 'moons', paramsA: { kernel: 0 }, paramsB: { kernel: 1 }, holdout: true },
	{ label: 'Single vs complete linkage', note: 'Press Play. Single linkage recovers both moons exactly. Complete linkage cuts across them, so only about 85% of samples land with their own moon.', a: 'hierarchical-clustering', b: 'hierarchical-clustering', dataset: 'moons', paramsA: { linkage: 0 }, paramsB: { linkage: 1 } },
	{ label: 'AdaBoost vs Gradient Boosting', note: 'Press Play. Stumps see one feature at a time, so AdaBoost ends at 50% Test accuracy. Depth 2 trees see both features together, and Gradient Boosting reaches 100%.', a: 'adaboost', b: 'gradient-boosting', dataset: 'xor', holdout: true },
	{ label: 'Logistic Regression vs MLP', note: 'Press Play. By epoch 3,000 the straight boundary is near 43% Test accuracy, and the network has wrapped around the inner circle for 100%.', a: 'logistic-regression', b: 'mlp', dataset: 'circles', holdout: true },
];
