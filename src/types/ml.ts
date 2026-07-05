/** A 2D observation. Coordinates live in the plot domain [-1, 1]. */
export interface DataPoint {
	x: number;
	y: number;
	/** Class index. Regression and unlabeled data use 0. */
	label: number;
}

export type Dataset = DataPoint[];

export type DatasetKind =
	| 'linear'
	| 'moons'
	| 'circles'
	| 'xor'
	| 'outliers'
	| 'trend'
	| 'curve'
	| 'trend-outliers'
	| 'blobs'
	| 'ellipse'
	| 'dirty'
	| 'wide'
	| 'imbalanced';

export type Hyperparameters = Record<string, number>;

/** Maps a location in the plot domain to a class index. */
export type Classifier = (x: number, y: number) => number;
