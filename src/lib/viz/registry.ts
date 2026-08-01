import type { AlgorithmId, AlgorithmSpec } from '../../types/viz';

/** Dynamic imports keep each docs page from downloading every algorithm. */
const loaders: Record<AlgorithmId, () => Promise<AlgorithmSpec>> = {
	'linear-regression': () => import('./specs/regression').then((m) => m.linearRegression),
	'polynomial-regression': () => import('./specs/regression').then((m) => m.polynomialRegression),
	'decision-tree': () => import('./specs/decisionTree').then((m) => m.decisionTree),
	'random-forest': () => import('./specs/ensembles').then((m) => m.randomForest),
	'gradient-boosting': () => import('./specs/ensembles').then((m) => m.gradientBoosting),
	adaboost: () => import('./specs/ensembles').then((m) => m.adaBoost),
	dbscan: () => import('./specs/clustering').then((m) => m.dbscan),
	knn: () => import('./specs/knn').then((m) => m.knn),
	svm: () => import('./specs/svm').then((m) => m.svm),
	'k-means': () => import('./specs/kmeans').then((m) => m.kMeans),
	pca: () => import('./specs/pca').then((m) => m.pca),
	perceptron: () => import('./specs/neural').then((m) => m.perceptron),
	mlp: () => import('./specs/neural').then((m) => m.mlp),
	'q-learning': () => import('./specs/rl').then((m) => m.qLearning),
	sarsa: () => import('./specs/rl').then((m) => m.sarsa),
	'deep-q-networks': () => import('./specs/dqn').then((m) => m.deepQNetwork),
	'proximal-policy-optimization': () => import('./specs/ppo').then((m) => m.proximalPolicyOptimization),
	'naive-bayes': () => import('./specs/statistical').then((m) => m.naiveBayes),
	'logistic-regression': () => import('./specs/statistical').then((m) => m.logisticRegression),
	'linear-discriminant-analysis': () => import('./specs/statistical').then((m) => m.lda),
	'gaussian-mixture-models': () => import('./specs/statistical').then((m) => m.gaussianMixture),
	'ridge-regression': () => import('./specs/statistical').then((m) => m.ridge),
	'lasso-regression': () => import('./specs/statistical').then((m) => m.lasso),
	'elastic-net': () => import('./specs/statistical').then((m) => m.elasticNet),
	'gaussian-process-regression': () => import('./specs/statistical').then((m) => m.gaussianProcess),
	'data-cleaning': () => import('./specs/preparation').then((m) => m.dataCleaning),
	'feature-scaling': () => import('./specs/preparation').then((m) => m.featureScaling),
	'data-splitting': () => import('./specs/preparation').then((m) => m.dataSplitting),
	'cross-validation': () => import('./specs/evaluation').then((m) => m.crossValidation),
	'hyperparameter-search': () => import('./specs/evaluation').then((m) => m.hyperparameterSearch),
	'threshold-metrics': () => import('./specs/evaluation').then((m) => m.thresholdMetrics),
};

export const ALGORITHM_IDS = Object.keys(loaders) as AlgorithmId[];

export const loadSpec = (id: AlgorithmId): Promise<AlgorithmSpec> => loaders[id]();
