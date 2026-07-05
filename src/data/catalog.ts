import type { DatasetKind } from '../types/ml';
import type { AlgorithmId } from '../types/viz';

export const FAMILIES = [
	'Data Preparation',
	'Validation',
	'Testing',
	'Supervised Learning',
	'Unsupervised Learning',
	'Reinforcement Learning',
	'Deep Learning',
] as const;

export type Family = (typeof FAMILIES)[number];

export interface CatalogEntry {
	name: string;
	family: Family;
	summary: string;
	/** Other names people search for. */
	aliases?: string;
	/** Present when the algorithm has a full page and sandbox on this site. */
	page?: { id: AlgorithmId; thumbnail: DatasetKind; path?: string };
	/** Wikipedia article title, for entries that have no page yet. */
	wiki?: string;
}

export const CATALOG: CatalogEntry[] = [
	{ name: 'Data Cleaning', family: 'Data Preparation', aliases: 'missing values imputation duplicates outliers IQR', summary: 'Find and treat missing values, duplicates and outliers.', page: { id: 'data-cleaning', thumbnail: 'dirty', path: 'data-preparation/cleaning' } },
	{ name: 'Data Processing', family: 'Data Preparation', aliases: 'feature scaling normalization standardization one-hot encoding', summary: 'Scale numbers and encode categories for algorithms.', page: { id: 'feature-scaling', thumbnail: 'wide', path: 'data-preparation/processing' } },
	{ name: 'Data Splitting', family: 'Data Preparation', aliases: 'train test validation stratified cross-validation k-fold leakage', summary: 'Separate training and test data to measure honestly.', page: { id: 'data-splitting', thumbnail: 'imbalanced', path: 'data-preparation/splitting' } },
	{ name: 'Cross-Validation', family: 'Validation', aliases: 'k-fold stratified hold-out leave-one-out nested', summary: 'Rotate the validation set for a more reliable score.', page: { id: 'cross-validation', thumbnail: 'outliers', path: 'validation/cross-validation' } },
	{ name: 'Hyperparameter Tuning', family: 'Validation', aliases: 'grid search random search bayesian optimization early stopping', summary: 'Choose model settings using validation scores.', page: { id: 'hyperparameter-search', thumbnail: 'moons', path: 'validation/hyperparameter-tuning' } },

	{ name: 'Classification Metrics', family: 'Testing', aliases: 'confusion matrix precision recall F1 ROC AUC accuracy', summary: 'Measure which kinds of mistakes a classifier makes.', page: { id: 'threshold-metrics', thumbnail: 'outliers', path: 'testing/classification-metrics' } },
	{ name: 'Regression Metrics', family: 'Testing', aliases: 'MAE RMSE R2 MAPE residuals', summary: 'Measure how far predictions land from real values.', page: { id: 'linear-regression', thumbnail: 'trend', path: 'testing/regression-metrics' } },

	{ name: 'Linear Regression', family: 'Supervised Learning', aliases: 'OLS least squares', summary: 'Fits a straight line by minimizing squared error.', page: { id: 'linear-regression', thumbnail: 'trend' } },
	{ name: 'Polynomial Regression', family: 'Supervised Learning', aliases: 'curve fitting', summary: 'Linear regression on powers of x, for curved trends.', page: { id: 'polynomial-regression', thumbnail: 'curve' } },
	{ name: 'Ridge Regression', family: 'Supervised Learning', aliases: 'L2 Tikhonov', summary: 'Least squares with an L2 penalty that shrinks coefficients.', wiki: 'Ridge_regression' },
	{ name: 'Lasso Regression', family: 'Supervised Learning', aliases: 'L1 sparse', summary: 'Least squares with an L1 penalty that drives coefficients to zero.', wiki: 'Lasso_(statistics)' },
	{ name: 'Elastic Net', family: 'Supervised Learning', aliases: 'L1 L2 regularization', summary: 'Blends the L1 and L2 penalties of lasso and ridge.', wiki: 'Elastic_net_regularization' },
	{ name: 'Logistic Regression', family: 'Supervised Learning', aliases: 'logit classification', summary: 'Models class probability with a sigmoid of a linear score.', wiki: 'Logistic_regression' },
	{ name: 'K-Nearest Neighbors', family: 'Supervised Learning', aliases: 'KNN instance-based lazy', summary: 'Classifies a point by the vote of the closest samples.', page: { id: 'knn', thumbnail: 'moons' } },
	{ name: 'Support Vector Machines', family: 'Supervised Learning', aliases: 'SVM SVC kernel SMO', summary: 'Finds the boundary with the widest margin between classes.', page: { id: 'svm', thumbnail: 'linear' } },
	{ name: 'Naive Bayes', family: 'Supervised Learning', aliases: 'bayes classifier', summary: 'Applies Bayes’ rule assuming features are independent given the class.', wiki: 'Naive_Bayes_classifier' },
	{ name: 'Decision Trees', family: 'Supervised Learning', aliases: 'CART gini', summary: 'Partitions the space with a hierarchy of axis-aligned questions.', page: { id: 'decision-tree', thumbnail: 'xor' } },
	{ name: 'Random Forest', family: 'Supervised Learning', aliases: 'bagging ensemble trees', summary: 'Averages many decorrelated trees trained on bootstrap samples.', wiki: 'Random_forest' },
	{ name: 'Gradient Boosting Machines', family: 'Supervised Learning', aliases: 'GBM GBDT ensemble', summary: 'Adds trees that each fit the gradient of the current loss.', wiki: 'Gradient_boosting' },
	{ name: 'XGBoost', family: 'Supervised Learning', aliases: 'gradient boosting ensemble', summary: 'Regularized gradient boosting engineered for speed and scale.', wiki: 'XGBoost' },
	{ name: 'LightGBM', family: 'Supervised Learning', aliases: 'gradient boosting histogram ensemble', summary: 'Histogram-based boosting that grows trees leaf by leaf.', wiki: 'LightGBM' },
	{ name: 'CatBoost', family: 'Supervised Learning', aliases: 'gradient boosting categorical ensemble', summary: 'Gradient boosting with built-in handling of categorical features.', wiki: 'CatBoost' },
	{ name: 'AdaBoost', family: 'Supervised Learning', aliases: 'adaptive boosting ensemble', summary: 'Reweights samples so each weak learner fixes earlier errors.', wiki: 'AdaBoost' },
	{ name: 'Extra Trees', family: 'Supervised Learning', aliases: 'extremely randomized trees ensemble', summary: 'A forest whose split thresholds are drawn at random.', wiki: 'Random_forest#ExtraTrees' },

	{ name: 'K-Means Clustering', family: 'Unsupervised Learning', aliases: 'Lloyd centroids', summary: 'Groups points around k centroids by alternating assign and update.', page: { id: 'k-means', thumbnail: 'blobs' } },
	{ name: 'K-Medians Clustering', family: 'Unsupervised Learning', aliases: 'manhattan robust', summary: 'A k-means variant that uses medians, making it robust to outliers.', wiki: 'K-medians_clustering' },
	{ name: 'DBSCAN', family: 'Unsupervised Learning', aliases: 'density clustering', summary: 'Grows clusters from dense regions and labels the rest as noise.', wiki: 'DBSCAN' },
	{ name: 'Hierarchical Clustering', family: 'Unsupervised Learning', aliases: 'agglomerative dendrogram', summary: 'Builds a tree of nested clusters by merging or splitting.', wiki: 'Hierarchical_clustering' },
	{ name: 'Mean Shift Clustering', family: 'Unsupervised Learning', aliases: 'mode seeking', summary: 'Moves points uphill on a density estimate until they meet at modes.', wiki: 'Mean_shift' },
	{ name: 'Gaussian Mixture Models', family: 'Unsupervised Learning', aliases: 'GMM EM soft clustering', summary: 'Soft clustering with Gaussians, fitted by expectation-maximization.', wiki: 'Mixture_model' },
	{ name: 'Principal Component Analysis', family: 'Unsupervised Learning', aliases: 'PCA dimensionality reduction eigenvectors', summary: 'Rotates data onto the directions of greatest variance.', page: { id: 'pca', thumbnail: 'ellipse' } },
	{ name: 't-Distributed Stochastic Neighbor Embedding', family: 'Unsupervised Learning', aliases: 't-SNE tsne embedding visualization', summary: 'Embeds data in 2D while preserving local neighborhoods.', wiki: 'T-distributed_stochastic_neighbor_embedding' },
	{ name: 'Uniform Manifold Approximation and Projection', family: 'Unsupervised Learning', aliases: 'UMAP embedding visualization', summary: 'Builds a neighbor graph and lays it out in low dimensions.', wiki: 'Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection' },
	{ name: 'Independent Component Analysis', family: 'Unsupervised Learning', aliases: 'ICA blind source separation', summary: 'Separates a signal into statistically independent sources.', wiki: 'Independent_component_analysis' },
	{ name: 'Linear Discriminant Analysis', family: 'Unsupervised Learning', aliases: 'LDA Fisher dimensionality reduction', summary: 'Projects data to maximize separation between class means. It uses labels, unlike the rest of this group.', wiki: 'Linear_discriminant_analysis' },
	{ name: 'Apriori Algorithm', family: 'Unsupervised Learning', aliases: 'association rules market basket', summary: 'Mines frequent itemsets and association rules from transactions.', wiki: 'Apriori_algorithm' },
	{ name: 'Eclat Algorithm', family: 'Unsupervised Learning', aliases: 'association rules frequent itemsets', summary: 'Finds frequent itemsets by intersecting transaction ID lists.', wiki: 'Association_rule_learning#Eclat_algorithm' },
	{ name: 'Isolation Forest', family: 'Unsupervised Learning', aliases: 'anomaly detection outlier', summary: 'Flags anomalies as the points random splits isolate fastest.', wiki: 'Isolation_forest' },
	{ name: 'One-Class Support Vector Machines', family: 'Unsupervised Learning', aliases: 'OCSVM novelty anomaly detection', summary: 'Learns a boundary around normal data to detect novelties.', wiki: 'One-class_classification' },

	{ name: 'Q-Learning', family: 'Reinforcement Learning', aliases: 'temporal difference tabular off-policy', summary: 'Learns the value of each action in each state from experience.', page: { id: 'q-learning', thumbnail: 'trend' } },
	{ name: 'State-Action-Reward-State-Action', family: 'Reinforcement Learning', aliases: 'SARSA on-policy TD', summary: 'On-policy temporal-difference control.', page: { id: 'sarsa', thumbnail: 'trend' } },
	{ name: 'Deep Q-Networks', family: 'Reinforcement Learning', aliases: 'DQN experience replay', summary: 'Q-learning with a neural network as the value function.', page: { id: 'deep-q-networks', thumbnail: 'trend' } },
	{ name: 'Deep Deterministic Policy Gradient', family: 'Reinforcement Learning', aliases: 'DDPG continuous control', summary: 'An off-policy actor-critic for continuous actions.', wiki: 'Actor-critic_algorithm' },
	{ name: 'Proximal Policy Optimization', family: 'Reinforcement Learning', aliases: 'PPO clipped objective', summary: 'Policy gradient with a clipped objective that keeps updates small.', wiki: 'Proximal_policy_optimization' },
	{ name: 'Asynchronous Advantage Actor-Critic', family: 'Reinforcement Learning', aliases: 'A3C parallel', summary: 'Parallel workers update a shared actor-critic asynchronously.', wiki: 'Actor-critic_algorithm' },
	{ name: 'Soft Actor-Critic', family: 'Reinforcement Learning', aliases: 'SAC maximum entropy', summary: 'An off-policy actor-critic that also maximizes policy entropy.', wiki: 'Actor-critic_algorithm' },

	{ name: 'Perceptron', family: 'Deep Learning', aliases: 'Rosenblatt linear classifier', summary: 'The original learning neuron: a linear boundary and a mistake-driven rule.', page: { id: 'perceptron', thumbnail: 'linear' } },
	{ name: 'Multilayer Perceptron', family: 'Deep Learning', aliases: 'MLP backpropagation', summary: 'Stacked nonlinear layers trained with backpropagation.', page: { id: 'mlp', thumbnail: 'moons' } },
	{ name: 'Feedforward Neural Networks', family: 'Deep Learning', aliases: 'FNN dense', summary: 'Networks whose connections only run from input to output.', wiki: 'Feedforward_neural_network' },
	{ name: 'Convolutional Neural Networks', family: 'Deep Learning', aliases: 'CNN convnet vision', summary: 'Share local filters across an image to learn visual features.', wiki: 'Convolutional_neural_network' },
	{ name: 'Recurrent Neural Networks', family: 'Deep Learning', aliases: 'RNN sequence', summary: 'Carry a hidden state through a sequence, step by step.', wiki: 'Recurrent_neural_network' },
	{ name: 'Long Short-Term Memory Networks', family: 'Deep Learning', aliases: 'LSTM gated', summary: 'Gated recurrent cells that retain information over long spans.', wiki: 'Long_short-term_memory' },
	{ name: 'Gated Recurrent Units', family: 'Deep Learning', aliases: 'GRU', summary: 'A simpler gated recurrent cell with fewer parameters than LSTM.', wiki: 'Gated_recurrent_unit' },
	{ name: 'Transformers', family: 'Deep Learning', aliases: 'attention GPT BERT LLM', summary: 'Relate all positions of a sequence through self-attention.', wiki: 'Transformer_(deep_learning)' },
	{ name: 'Autoencoders', family: 'Deep Learning', aliases: 'representation compression', summary: 'Learn a compressed code by reconstructing their own input.', wiki: 'Autoencoder' },
	{ name: 'Variational Autoencoders', family: 'Deep Learning', aliases: 'VAE generative latent', summary: 'Autoencoders whose latent space is a probability distribution.', wiki: 'Variational_autoencoder' },
	{ name: 'Generative Adversarial Networks', family: 'Deep Learning', aliases: 'GAN generator discriminator', summary: 'A generator and a discriminator trained against each other.', wiki: 'Generative_adversarial_network' },
	{ name: 'Diffusion Models', family: 'Deep Learning', aliases: 'denoising generative image', summary: 'Generate data by learning to reverse gradual noising.', wiki: 'Diffusion_model' },
	{ name: 'Graph Neural Networks', family: 'Deep Learning', aliases: 'GNN message passing', summary: 'Pass messages along edges to learn from graph data.', wiki: 'Graph_neural_network' },
	{ name: 'Siamese Networks', family: 'Deep Learning', aliases: 'similarity metric learning contrastive', summary: 'Twin networks with shared weights that compare two inputs.', wiki: 'Siamese_neural_network' },
	{ name: 'Restricted Boltzmann Machines', family: 'Deep Learning', aliases: 'RBM energy-based', summary: 'Two-layer energy-based models of a data distribution.', wiki: 'Restricted_Boltzmann_machine' },
];

const GUIDE_FAMILIES: Family[] = ['Data Preparation', 'Validation', 'Testing'];

export const ALGORITHM_COUNT = CATALOG.filter((e) => !GUIDE_FAMILIES.includes(e.family)).length;

export const FEATURED = CATALOG.filter((e) => e.page);
