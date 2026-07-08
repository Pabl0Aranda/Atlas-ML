import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { choleskyFactor, fitGp, gpPredict } from './gaussianProcess';
import { ellipse, gmmStep, initGmm } from './gmm';
import { accuracy } from './knn';
import { fitLda, ldaScore } from './lda';
import { initLogistic, logisticLoss, logisticProbability, logisticStep } from './logisticRegression';
import { classifyNaiveBayes, fitNaiveBayes } from './naiveBayes';
import { initRegularized, penalizedLoss, regularizedStep, softThreshold, type PenaltyParams } from './regularizedRegression';

const repeat = <S>(n: number, s: S, f: (s: S) => S) => {
	for (let i = 0; i < n; i++) s = f(s);
	return s;
};

describe('naive Bayes', () => {
	it('estimates priors and means', () => {
		const m = fitNaiveBayes([{ x: 0, y: 0, label: 0 }, { x: 2, y: 2, label: 0 }, { x: 5, y: 5, label: 1 }]);
		expect(m[0].prior).toBeCloseTo(2 / 3);
		expect(m[0].mean.x).toBe(1);
		expect(m[0].variance.x).toBe(1);
	});
	it('falls back to the majority class with no features', () => {
		const data = generateDataset('imbalanced');
		expect(classifyNaiveBayes(fitNaiveBayes(data), 0.9, 0.9, [])).toBe(0);
	});
	it('classifies separable and curved data', () => {
		for (const [kind, floor] of [['linear', 0.97], ['circles', 0.95]] as const) {
			const data = generateDataset(kind);
			const m = fitNaiveBayes(data);
			expect(accuracy(data, (x, y) => classifyNaiveBayes(m, x, y))).toBeGreaterThan(floor);
		}
	});
});

describe('logistic regression', () => {
	const train = (kind: 'linear' | 'circles', degree: number) => {
		const data = generateDataset(kind);
		const s = repeat(1500, initLogistic(data, degree), (s) => logisticStep(s, data, degree, 2));
		return { s, score: accuracy(data, (x, y) => (logisticProbability(s.weights, x, y, degree) > 0.5 ? 1 : 0)) };
	};
	it('starts at log 2 and decreases the loss', () => {
		const { s } = train('linear', 1);
		expect(s.lossHistory[0]).toBeCloseTo(Math.LN2);
		s.lossHistory.slice(1).forEach((l, i) => expect(l).toBeLessThanOrEqual(s.lossHistory[i] + 1e-9));
	});
	it('solves circles only with quadratic features', () => {
		expect(train('circles', 1).score).toBeLessThan(0.7);
		expect(train('circles', 2).score).toBeGreaterThan(0.97);
	});
	it('matches a numeric gradient', () => {
		const data = generateDataset('moons', 20);
		const w = [0.1, -0.2, 0.3];
		const after = logisticStep({ weights: w, epoch: 0, loss: 0, lossHistory: [] }, data, 1, 1, 0.1).weights;
		const h = 1e-6;
		const numeric = (logisticLoss([w[0], w[1] + h, w[2]], data, 1, 0.1) - logisticLoss([w[0], w[1] - h, w[2]], data, 1, 0.1)) / (2 * h);
		expect(w[1] - after[1]).toBeCloseTo(numeric, 5);
	});
});

describe('LDA', () => {
	it('separates linear data but not circles', () => {
		const score = (kind: 'linear' | 'circles') => {
			const data = generateDataset(kind);
			const m = fitLda(data);
			return accuracy(data, (x, y) => (ldaScore(m, x, y) > 0 ? 1 : 0));
		};
		expect(score('linear')).toBeGreaterThan(0.97);
		expect(score('circles')).toBeLessThan(0.7);
	});
	it('points from class 0 toward class 1', () => {
		const m = fitLda(generateDataset('linear'));
		expect(ldaScore(m, m.means[1].x, m.means[1].y)).toBeGreaterThan(ldaScore(m, m.means[0].x, m.means[0].y));
	});
});

describe('GMM', () => {
	const data = generateDataset('blobs');
	const s = repeat(100, initGmm(data, 4), (s) => (s.converged ? s : gmmStep(s, data)));
	it('never decreases the likelihood and converges', () => {
		s.history.slice(1).forEach((l, i) => expect(l).toBeGreaterThanOrEqual(s.history[i] - 1e-6));
		expect(s.converged).toBe(true);
	});
	it('keeps valid weights and responsibilities', () => {
		expect(s.components.reduce((a, c) => a + c.weight, 0)).toBeCloseTo(1);
		for (const row of s.responsibilities) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
	});
	it('fits a correlated cluster better than a round one', () => {
		const cloud = generateDataset('ellipse');
		const fit = gmmStep(initGmm(cloud, 1), cloud);
		expect(fit.components[0].cov[1]).toBeGreaterThan(0.03);
		expect(ellipse(fit.components[0])).toHaveLength(49);
	});
});

describe('regularized regression', () => {
	const data = generateDataset('curve');
	const fit = (penalty: PenaltyParams) =>
		repeat(800, initRegularized(data, 9, penalty), (s) => regularizedStep(s, data, penalty));
	const nonZero = (w: number[]) => w.slice(1).filter((v) => v !== 0).length;

	it('soft-thresholds', () => {
		expect(softThreshold(3, 1)).toBe(2);
		expect(softThreshold(-3, 1)).toBe(-2);
		expect(softThreshold(0.5, 1) === 0).toBe(true);
	});
	it('never increases the objective', () => {
		const s = fit({ lambda: 0.01, l1Ratio: 0.5 });
		s.lossHistory.slice(1).forEach((l, i) => expect(l).toBeLessThanOrEqual(s.lossHistory[i] + 1e-9));
	});
	it('lasso zeroes coefficients while ridge keeps them all', () => {
		expect(nonZero(fit({ lambda: 0.01, l1Ratio: 1 }).weights)).toBeLessThan(6);
		expect(nonZero(fit({ lambda: 0.01, l1Ratio: 0 }).weights)).toBe(9);
	});
	it('a larger penalty shrinks harder', () => {
		const size = (lambda: number) => fit({ lambda, l1Ratio: 0 }).weights.slice(1).reduce((s, w) => s + w * w, 0);
		expect(size(0.1)).toBeLessThan(size(0.001));
	});
	it('reports the same loss it optimizes', () => {
		const penalty = { lambda: 0.01, l1Ratio: 0.5 };
		const s = fit(penalty);
		expect(s.loss).toBeCloseTo(penalizedLoss(s.weights, data, penalty));
	});
});

describe('Gaussian process', () => {
	const params = { lengthScale: 0.3, noise: 0.01 };
	it('factors a matrix', () => {
		expect(choleskyFactor([[4, 2], [2, 5]])).toEqual([[2, 0], [1, 2]]);
	});
	it('interpolates observations and is certain near them', () => {
		const model = fitGp([-0.5, 0, 0.5], [1, -1, 1], params);
		expect(gpPredict(model, 0).mean).toBeCloseTo(-1, 2);
		expect(gpPredict(model, 0).variance).toBeLessThan(0.01);
	});
	it('reverts to the prior far from the data', () => {
		const far = gpPredict(fitGp([0], [1], params), 5);
		expect(far.mean).toBeCloseTo(0);
		expect(far.variance).toBeCloseTo(1);
	});
});
