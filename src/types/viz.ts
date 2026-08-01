import type { Classifier, Dataset, DatasetKind, Hyperparameters } from './ml';

export type AlgorithmId =
	| 'linear-regression'
	| 'polynomial-regression'
	| 'decision-tree'
	| 'random-forest'
	| 'gradient-boosting'
	| 'adaboost'
	| 'dbscan'
	| 'knn'
	| 'svm'
	| 'k-means'
	| 'pca'
	| 'perceptron'
	| 'mlp'
	| 'q-learning'
	| 'sarsa'
	| 'deep-q-networks'
	| 'proximal-policy-optimization'
	| 'naive-bayes'
	| 'logistic-regression'
	| 'linear-discriminant-analysis'
	| 'gaussian-mixture-models'
	| 'ridge-regression'
	| 'lasso-regression'
	| 'elastic-net'
	| 'gaussian-process-regression'
	| 'data-cleaning'
	| 'feature-scaling'
	| 'data-splitting'
	| 'cross-validation'
	| 'hyperparameter-search'
	| 'threshold-metrics';

export interface ParamSpec {
	key: string;
	label: string;
	value: number;
	min?: number;
	max?: number;
	step?: number;
	/** The run does not depend on this value, so changing it redraws without restarting. */
	live?: boolean;
	/** Renders a select instead of a slider. */
	options?: { label: string; value: number }[];
}

export interface ScenePoint {
	x: number;
	y: number;
	cls: number;
	/** Draws a ring, e.g. around support vectors. */
	ring?: boolean;
	/** Size relative to a normal point, e.g. a sample's boosting weight. */
	scale?: number;
	/** Draws in the neutral color, for points with no class such as noise. */
	muted?: boolean;
}

export interface SceneLine {
	path: [number, number][];
	tone: 'ink' | 'muted';
	dashed?: boolean;
}

export interface GridCell {
	tile: 'free' | 'wall' | 'cliff' | 'start' | 'goal';
	/** Normalized learned value from 0 to 1. */
	value: number;
	/** Best action, ordered up, right, down, left. */
	arrow?: number;
}

/** A gridworld, drawn in place of the 2D plot. Cells are indexed row by row. */
export interface SceneGrid {
	cols: number;
	rows: number;
	cells: GridCell[];
	agent: number;
	path: number[];
}

/** Everything a plot shows, independent of how it is rendered. */
export interface Scene {
	region?: Classifier;
	points: ScenePoint[];
	lines?: SceneLine[];
	/** Large markers such as centroids. */
	markers?: { x: number; y: number; cls: number }[];
	grid?: SceneGrid;
}

export interface DiagramNode {
	x: number;
	y: number;
	label?: string;
	/** Class color. Omit for a neutral node. */
	cls?: number;
	/** Fill strength from 0 to 1. */
	strength?: number;
}

/** Node-link figure in a unit square: a decision tree or a network. */
export interface Diagram {
	title: string;
	nodes: DiagramNode[];
	edges: { from: number; to: number; weight?: number }[];
}

export interface Metric {
	label: string;
	value: string;
}

export interface Series {
	label: string;
	values: number[];
}

export interface AlgorithmSpec<S = any> {
	datasets: DatasetKind[];
	params: ParamSpec[];
	/** False for regression, where the click position alone defines a sample. */
	labeled: boolean;
	/** False when clicking the plot should not add samples. Defaults to true. */
	editable?: boolean;
	/** Steps per animation tick while playing. */
	speed: number;
	/** Steps used for the static catalog thumbnail. */
	thumbnailSteps: number;
	legend(params: Hyperparameters): string[];
	init(data: Dataset, params: Hyperparameters): S;
	step(state: S, data: Dataset, params: Hyperparameters): S;
	isDone(state: S, params: Hyperparameters): boolean;
	scene(state: S, data: Dataset, params: Hyperparameters): Scene;
	metrics(state: S, data: Dataset, params: Hyperparameters): Metric[];
	caption(state: S, data: Dataset, params: Hyperparameters): string;
	series?(state: S): Series;
	diagram?(state: S, probe: { x: number; y: number }): Diagram | undefined;
	/** Text shown for the location under the cursor. */
	describe?(state: S, probe: { x: number; y: number }, params: Hyperparameters): string;
}
