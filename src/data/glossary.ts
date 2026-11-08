export interface GlossaryEntry {
	term: string;
	/** One or two plain sentences, short enough for a tooltip. */
	definition: string;
	/** Other spellings found in the text, such as plurals. Matching ignores case. */
	forms: string[];
	/** A page that explains the term in depth. */
	see?: string;
}

/** Machine learning jargon, defined once and shown wherever a page uses it. */
export const GLOSSARY = {
	accuracy: {
		term: 'Accuracy',
		definition: 'The share of predictions that are correct. It can mislead when one class is much more common than the others.',
		forms: ['accuracy'],
		see: '/testing/classification-metrics/',
	},
	bias: {
		term: 'Bias',
		definition: 'Error from a model that is too simple to capture the real pattern, so it misses in the same direction every time.',
		forms: ['bias'],
		see: '/concepts/bias-variance/',
	},
	centroid: {
		term: 'Centroid',
		definition: 'The center of a cluster: the average position of the points assigned to it.',
		forms: ['centroid', 'centroids'],
		see: '/algorithms/k-means/',
	},
	classification: {
		term: 'Classification',
		definition: 'Predicting which category, or class, a sample belongs to, such as spam or not spam.',
		forms: ['classification', 'classifier', 'classifiers'],
	},
	cluster: {
		term: 'Cluster',
		definition: 'A group of points that are closer to each other than to points in other groups, found without labels.',
		forms: ['cluster', 'clusters', 'clustering'],
		see: '/algorithms/k-means/',
	},
	'cross-validation': {
		term: 'Cross-validation',
		definition: 'Splitting the data into several folds and training several times, holding out a different fold each time, for a steadier score.',
		forms: ['cross-validation'],
		see: '/validation/cross-validation/',
	},
	'decision-boundary': {
		term: 'Decision boundary',
		definition: 'The line or surface where a classifier switches from predicting one class to another.',
		forms: ['decision boundary', 'decision boundaries'],
	},
	ensemble: {
		term: 'Ensemble',
		definition: 'A model that combines the predictions of many simpler models, usually to be more accurate than any of them alone.',
		forms: ['ensemble', 'ensembles'],
		see: '/algorithms/random-forest/',
	},
	epoch: {
		term: 'Epoch',
		definition: 'One full pass of the training algorithm over the whole training set.',
		forms: ['epoch', 'epochs'],
	},
	feature: {
		term: 'Feature',
		definition: 'One measured input of a sample, such as age or price. A model learns from features to predict the label.',
		forms: ['feature', 'features'],
		see: '/data-preparation/processing/',
	},
	gradient: {
		term: 'Gradient',
		definition: 'The direction in which the loss rises fastest. Training steps the opposite way to reduce the loss.',
		forms: ['gradient', 'gradients'],
		see: '/concepts/gradient-descent/',
	},
	'held-out-data': {
		term: 'Held-out data',
		definition: 'Samples kept away from training, so they can show how the model does on data it has never seen.',
		forms: ['held-out data', 'held-out', 'held out'],
		see: '/data-preparation/splitting/',
	},
	hyperparameter: {
		term: 'Hyperparameter',
		definition: 'A setting chosen before training rather than learned from the data, such as k in KNN or the learning rate.',
		forms: ['hyperparameter', 'hyperparameters'],
		see: '/validation/hyperparameter-tuning/',
	},
	inertia: {
		term: 'Inertia',
		definition: 'The sum of squared distances from each point to its centroid. Lower means tighter clusters.',
		forms: ['inertia'],
		see: '/algorithms/k-means/',
	},
	kernel: {
		term: 'Kernel',
		definition: 'A function that measures similarity between samples, letting a linear method draw curved boundaries.',
		forms: ['kernel', 'kernels'],
		see: '/algorithms/svm/',
	},
	label: {
		term: 'Label',
		definition: 'The answer a model should learn to predict for a sample, such as its class or its price.',
		forms: ['label', 'labels', 'labeled'],
	},
	'learning-rate': {
		term: 'Learning rate',
		definition: 'How big a step training takes on each update. Too small is slow, and too large can overshoot and never settle.',
		forms: ['learning rate', 'learning rates'],
		see: '/concepts/gradient-descent/',
	},
	loss: {
		term: 'Loss function',
		definition: 'A number that measures how wrong the model is on the training data. Training tries to make it as small as possible.',
		forms: ['loss function', 'loss functions', 'loss'],
		see: '/concepts/gradient-descent/',
	},
	outlier: {
		term: 'Outlier',
		definition: 'A sample far away from the rest, from a rare case or a measurement error. It can pull some models off course.',
		forms: ['outlier', 'outliers'],
		see: '/data-preparation/cleaning/',
	},
	overfitting: {
		term: 'Overfitting',
		definition: 'When a model memorizes the training data, noise included, so it scores well in training but poorly on new data.',
		forms: ['overfitting', 'overfit', 'overfits'],
		see: '/concepts/overfitting-underfitting/',
	},
	policy: {
		term: 'Policy',
		definition: 'In reinforcement learning, the rule an agent follows to choose an action in each state.',
		forms: ['policy', 'policies'],
		see: '/algorithms/q-learning/',
	},
	precision: {
		term: 'Precision',
		definition: 'Of the samples the model called positive, the share that really are positive.',
		forms: ['precision'],
		see: '/testing/classification-metrics/',
	},
	recall: {
		term: 'Recall',
		definition: 'Of the samples that really are positive, the share the model found.',
		forms: ['recall'],
		see: '/testing/classification-metrics/',
	},
	regression: {
		term: 'Regression',
		definition: 'Predicting a number, such as a price or a temperature, rather than a category.',
		forms: ['regression'],
		see: '/algorithms/linear-regression/',
	},
	regularization: {
		term: 'Regularization',
		definition: 'A penalty on model complexity, such as large weights, that trades a little training accuracy for better results on new data.',
		forms: ['regularization', 'regularized'],
		see: '/algorithms/ridge-regression/',
	},
	reward: {
		term: 'Reward',
		definition: 'In reinforcement learning, the score an agent receives after an action. It learns to collect as much as possible over time.',
		forms: ['reward', 'rewards'],
		see: '/algorithms/q-learning/',
	},
	'test-set': {
		term: 'Test set',
		definition: 'Data used once, at the very end, to measure how well the final model works on unseen samples.',
		forms: ['test set', 'test sets'],
		see: '/data-preparation/splitting/',
	},
	'training-set': {
		term: 'Training set',
		definition: 'The data a model learns from.',
		forms: ['training set', 'training sets', 'training data'],
		see: '/data-preparation/splitting/',
	},
	underfitting: {
		term: 'Underfitting',
		definition: 'When a model is too simple to capture the pattern, so it scores poorly on training and new data alike.',
		forms: ['underfitting', 'underfit', 'underfits'],
		see: '/concepts/overfitting-underfitting/',
	},
	'validation-set': {
		term: 'Validation set',
		definition: 'Data held out during training to compare models and tune hyperparameters, kept separate from the test set.',
		forms: ['validation set', 'validation sets'],
		see: '/validation/overview/',
	},
	variance: {
		term: 'Variance',
		definition: 'Error from a model that is too sensitive to its training sample, so it changes a lot when the data changes.',
		forms: ['variance'],
		see: '/concepts/bias-variance/',
	},
} satisfies Record<string, GlossaryEntry>;

export type GlossaryId = keyof typeof GLOSSARY;

/** Terms in alphabetical order, for the glossary page. */
export const GLOSSARY_ENTRIES = (Object.entries(GLOSSARY) as [GlossaryId, GlossaryEntry][]).sort((a, b) =>
	a[1].term.localeCompare(b[1].term),
);
