import type { Dataset, Hyperparameters } from '../../../types/ml';
import type { AlgorithmSpec, SceneLine, ScenePoint } from '../../../types/viz';
import { generateDataset } from '../../ml/datasets';
import { classWeights, MINORITY, minorityScores, oversample, smote, undersample, weightedLogisticStep } from '../../ml/imbalance';
import { logisticProbability } from '../../ml/logisticRegression';
import { percent } from '../scene';

export const METHODS = ['No rebalancing', 'Undersample', 'Oversample', 'SMOTE', 'Class weights'];
const [NONE, UNDER, OVER, SMOTE, WEIGHTS] = [0, 1, 2, 3, 4];

export const EPOCHS = 300;
const RATE = 2;

/**
 * A separate, larger sample from the same source, never resampled. Rebalancing is a change
 * to training data only: the test set has to keep the real class balance to mean anything.
 */
export const TEST_SET: Dataset = generateDataset('imbalanced', 400, 11);

export interface ImbalanceState {
	/** 0 shows the raw training data, 1 the rebalanced data, and 2 onward the training. */
	stage: number;
	weights: number[];
	epoch: number;
}

interface Prepared {
	/** What the model trains on. */
	train: Dataset;
	classWeight: [number, number];
	points: ScenePoint[];
	lines: SceneLine[];
}

const count = (data: Dataset, label: number) => data.filter((p) => p.label === label).length;

export function prepare(data: Dataset, p: Hyperparameters): Prepared {
	const real = data.map((d) => ({ x: d.x, y: d.y, cls: d.label }));
	switch (p.method) {
		case UNDER: {
			const { kept, removed } = undersample(data);
			return { train: kept, classWeight: [1, 1], points: [...removed.map((d) => ({ x: d.x, y: d.y, cls: d.label, muted: true })), ...kept.map((d) => ({ x: d.x, y: d.y, cls: d.label }))], lines: [] };
		}
		case OVER: {
			const copies = oversample(data);
			// Copies land exactly on their originals, so ring each copied sample, larger the more copies it has.
			const times = (d: Dataset[number]) => copies.filter((c) => c.x === d.x && c.y === d.y).length;
			return {
				train: [...data, ...copies],
				classWeight: [1, 1],
				points: data.map((d) => {
					const n = d.label === MINORITY ? times(d) : 0;
					return n ? { x: d.x, y: d.y, cls: d.label, ring: true, scale: Math.min(2.4, 1 + 0.2 * n) } : { x: d.x, y: d.y, cls: d.label };
				}),
				lines: [],
			};
		}
		case SMOTE: {
			const synthetic = smote(data, p.k);
			return {
				train: [...data, ...synthetic.map((s) => s.point)],
				classWeight: [1, 1],
				points: [...real, ...synthetic.map((s) => ({ x: s.point.x, y: s.point.y, cls: MINORITY, hollow: true }))],
				lines: synthetic.map((s) => ({ path: [[s.from.x, s.from.y], [s.to.x, s.to.y]], tone: 'muted', dashed: true })),
			};
		}
		case WEIGHTS: {
			const classWeight = classWeights(data);
			const ratio = classWeight[MINORITY] / (classWeight[1 - MINORITY] || 1);
			return {
				train: data,
				classWeight,
				points: real.map((d) => (d.cls === MINORITY ? { ...d, scale: Math.min(2.4, Math.sqrt(ratio)) } : d)),
				lines: [],
			};
		}
		default:
			return { train: data, classWeight: [1, 1], points: real, lines: [] };
	}
}

const classify = (weights: number[]) => (x: number, y: number) => (logisticProbability(weights, x, y, 1) > 0.5 ? 1 : 0);

export const imbalancedData: AlgorithmSpec<ImbalanceState> = {
	datasets: ['imbalanced'],
	params: [
		{ key: 'method', label: 'Rebalancing', value: SMOTE, options: METHODS.map((label, value) => ({ label, value })) },
		{ key: 'k', label: 'SMOTE neighbors k', value: 5, min: 1, max: 10, step: 1 },
	],
	labeled: true,
	speed: 10,
	thumbnailSteps: EPOCHS + 2,
	legend: () => ['Common class', 'Rare class'],
	init: () => ({ stage: 0, weights: [0, 0, 0], epoch: 0 }),
	step(s, data, p) {
		if (s.stage < 2) return { ...s, stage: s.stage + 1 };
		if (s.epoch >= EPOCHS) return s;
		const { train, classWeight } = prepare(data, p);
		return { ...s, weights: weightedLogisticStep(s.weights, train, classWeight, RATE), epoch: s.epoch + 1 };
	},
	isDone: (s) => s.stage >= 2 && s.epoch >= EPOCHS,
	scene(s, data, p) {
		if (s.stage === 0) return { points: data.map((d) => ({ x: d.x, y: d.y, cls: d.label })) };
		const { points, lines } = prepare(data, p);
		return { points, lines, region: s.epoch > 0 ? classify(s.weights) : undefined };
	},
	metrics(s, data, p) {
		const train = s.stage === 0 ? data : prepare(data, p).train;
		const scores = s.epoch > 0 ? minorityScores(classify(s.weights), TEST_SET) : undefined;
		const show = (v: number | undefined) => (v === undefined ? '–' : percent(v));
		return [
			{ label: 'Training: common / rare', value: `${count(train, 1 - MINORITY)} / ${count(train, MINORITY)}` },
			{ label: 'Test accuracy', value: show(scores?.accuracy) },
			{ label: 'Rare class recall', value: show(scores?.recall) },
			{ label: 'Rare class precision', value: show(scores?.precision) },
		];
	},
	caption(s, data, p) {
		const rare = count(data, MINORITY);
		if (s.stage === 0) return `${rare} of ${data.length} training samples are rare. A model that always guesses the common class is already right about ${percent(1 - rare / Math.max(data.length, 1))} of the time. Step to rebalance.`;
		if (s.stage === 1) {
			switch (p.method) {
				case UNDER: return 'Undersampling dropped the gray common samples at random, keeping as many as there are rare ones. Balanced, but most of the data is gone.';
				case OVER: return 'Oversampling copied rare samples at random. Copies sit on top of their originals, so the rings mark which were copied, larger for more copies.';
				case SMOTE: return 'SMOTE placed each hollow sample somewhere on a dashed segment between two nearby rare samples, filling in the space between them.';
				case WEIGHTS: return 'Class weights keep the data as it is. Each rare sample now counts more in the loss, drawn larger here, so both classes weigh the same in total.';
				default: return 'No rebalancing: the model trains on the data as it is. Step to train a logistic regression.';
			}
		}
		if (s.epoch < EPOCHS) return `Training a logistic regression, epoch ${s.epoch} of ${EPOCHS}. The scores are measured on ${TEST_SET.length} separate test samples that keep the real balance.`;
		return 'Done. Compare the rare class recall and precision with No rebalancing: catching more rare samples usually means more false alarms too.';
	},
};
