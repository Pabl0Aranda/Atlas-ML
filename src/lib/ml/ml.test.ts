import { describe, expect, it } from 'vitest';
import { buildTree, classifyTree, gini, treeDepth } from './decisionTree';
import { generateDataset } from './datasets';
import { accuracy, knnClassify, knnLeaveOneOut } from './knn';
import { initKMeans, kMeansStep } from './kmeans';
import { forward, initMlp, mlpLoss, mlpProbability, mlpStep } from './mlp';
import { eigenvalues2d, initPca, pcaStep } from './pca';
import { initPerceptron, perceptronClassify, perceptronEpoch } from './perceptron';
import { createRng } from './random';
import { initRegression, predict, regressionLoss, regressionStep } from './regression';
import { initSvm, supportVectorCount, svmDecision, svmStep, type SvmParams } from './svm';

const repeat = <S>(n: number, s: S, f: (s: S) => S) => {
	for (let i = 0; i < n; i++) s = f(s);
	return s;
};

describe('datasets', () => {
	it('are deterministic, balanced and in range', () => {
		expect(createRng(1)()).toBe(createRng(1)());
		for (const kind of ['linear', 'moons', 'circles', 'blobs', 'curve'] as const) {
			const data = generateDataset(kind);
			expect(data).toEqual(generateDataset(kind));
			expect(data.every((p) => Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1)).toBe(true);
		}
		expect(generateDataset('moons').filter((p) => p.label === 1)).toHaveLength(60);
	});
});

describe('regression', () => {
	it('recovers a noiseless line', () => {
		const data = [-1, -0.5, 0, 0.5, 1].map((x) => ({ x, y: 2 * x + 0.5, label: 0 }));
		const s = repeat(2000, initRegression(data, 1), (s) => regressionStep(s, data, 0.1));
		expect(s.weights[0]).toBeCloseTo(0.5, 4);
		expect(s.weights[1]).toBeCloseTo(2, 4);
	});
	it('decreases loss monotonically at a safe learning rate', () => {
		const data = generateDataset('curve');
		const s = repeat(100, initRegression(data, 3), (s) => regressionStep(s, data, 0.1));
		s.lossHistory.slice(1).forEach((l, i) => expect(l).toBeLessThanOrEqual(s.lossHistory[i] + 1e-12));
	});
	it('matches a numeric gradient', () => {
		const data = generateDataset('curve', 20);
		const w = [0.1, -0.3, 0.2];
		const l2 = 0.05;
		const after = regressionStep({ weights: w, epoch: 0, loss: 0, lossHistory: [] }, data, 1, l2).weights;
		w.forEach((_, j) => {
			const h = 1e-6;
			const up = w.map((v, i) => (i === j ? v + h : v));
			const down = w.map((v, i) => (i === j ? v - h : v));
			const numeric = (regressionLoss(up, data, l2) - regressionLoss(down, data, l2)) / (2 * h);
			expect(w[j] - after[j]).toBeCloseTo(numeric, 5);
		});
	});
	it('evaluates polynomials', () => expect(predict([1, 2, 3], 2)).toBe(17));
});

describe('decision tree', () => {
	it('computes gini', () => {
		expect(gini([{ x: 0, y: 0, label: 0 }, { x: 0, y: 0, label: 1 }])).toBeCloseTo(0.5);
	});
	it('fits XOR when deep and respects maxDepth', () => {
		const data = generateDataset('xor');
		const tree = buildTree(data, 8);
		expect(accuracy(data, (x, y) => classifyTree(tree, x, y))).toBe(1);
		expect(treeDepth(buildTree(data, 2))).toBeLessThanOrEqual(2);
	});
});

describe('knn', () => {
	it('is perfect on training data at k=1 and separates moons out of sample', () => {
		const data = generateDataset('moons');
		expect(accuracy(data, (x, y) => knnClassify(data, 1, x, y))).toBe(1);
		expect(knnLeaveOneOut(data, 5)).toBeGreaterThan(0.95);
	});
});

describe('svm', () => {
	const linear: SvmParams = { C: 1, kernel: 'linear', gamma: 1 };
	const train = (data: ReturnType<typeof generateDataset>, p: SvmParams) =>
		repeat(40, initSvm(data), (s) => svmStep(s, data, p));
	it('separates linear data with few support vectors and keeps the dual constraint', () => {
		const data = generateDataset('linear');
		const s = train(data, linear);
		expect(accuracy(data, (x, y) => (svmDecision(s, data, linear, x, y) > 0 ? 1 : 0))).toBeGreaterThan(0.97);
		expect(supportVectorCount(s)).toBeLessThan(40);
		const balance = s.alphas.reduce((sum, a, i) => sum + a * (data[i].label ? 1 : -1), 0);
		expect(balance).toBeCloseTo(0, 6);
		expect(Math.max(...s.alphas)).toBeLessThanOrEqual(linear.C + 1e-9);
	});
	it('solves circles with the RBF kernel', () => {
		const data = generateDataset('circles');
		const rbf: SvmParams = { C: 5, kernel: 'rbf', gamma: 5 };
		const s = train(data, rbf);
		expect(accuracy(data, (x, y) => (svmDecision(s, data, rbf, x, y) > 0 ? 1 : 0))).toBeGreaterThan(0.97);
	});
});

describe('k-means', () => {
	it('never increases inertia and converges on blobs', () => {
		const data = generateDataset('blobs');
		const s = repeat(30, initKMeans(data, 4), (s) => kMeansStep(s, data));
		s.inertiaHistory.slice(1).forEach((v, i) => expect(v).toBeLessThanOrEqual(s.inertiaHistory[i] + 1e-12));
		expect(s.converged).toBe(true);
	});
});

describe('pca', () => {
	it('converges to the top eigenvalue share', () => {
		const data = generateDataset('ellipse');
		const s = repeat(100, initPca(data), pcaStep);
		const [l1, l2] = eigenvalues2d(s.covariance);
		expect(s.explained).toBeCloseTo(l1 / (l1 + l2), 6);
		expect(s.explained).toBeGreaterThan(0.85);
		expect(Math.hypot(...s.axis)).toBeCloseTo(1);
	});
});

describe('perceptron', () => {
	it('reaches zero mistakes on separable data', () => {
		const data = generateDataset('linear').filter((p) => (p.label ? p.x + p.y > 0.2 : p.x + p.y < -0.2));
		const s = repeat(200, initPerceptron(), (s) => perceptronEpoch(s, data, 0.1));
		expect(s.mistakes).toBe(0);
		expect(accuracy(data, (x, y) => perceptronClassify(s, x, y))).toBe(1);
	});
});

describe('mlp', () => {
	it('learns XOR, which a perceptron cannot', () => {
		const data = generateDataset('xor');
		const s = repeat(1500, initMlp(data, [8, 8]), (s) => mlpStep(s, data, 0.5));
		expect(s.loss).toBeLessThan(s.lossHistory[0] / 2);
		expect(accuracy(data, (x, y) => (mlpProbability(s, x, y) > 0.5 ? 1 : 0))).toBeGreaterThan(0.93);
	});
	it('matches a numeric gradient', () => {
		const data = generateDataset('moons', 10);
		const s = initMlp(data, [3]);
		const after = mlpStep(s, data, 1);
		const h = 1e-6;
		const shifted = (d: number) => {
			const weights = structuredClone(s.weights);
			weights[0][1][0] += d;
			return mlpLoss({ weights, biases: s.biases }, data);
		};
		expect(s.weights[0][1][0] - after.weights[0][1][0]).toBeCloseTo((shifted(h) - shifted(-h)) / (2 * h), 5);
		expect(forward(s, 0, 0).map((l) => l.length)).toEqual([2, 3, 1]);
	});
});
