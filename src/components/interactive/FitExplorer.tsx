import { useMemo, useState } from 'preact/hooks';
import { FIT_NEIGHBORS, fitVerdict, knnFitCurve, sampleSplit, type FitPoint, type FitVerdict } from '@/lib/ml/fitCurve';
import { knnClassify } from '@/lib/ml/knn';
import { withHeldOut } from '@/lib/viz/sandboxSetup';
import { classPoints, percent } from '@/lib/viz/scene';
import { fieldLabelClass, MetricList, PlotLegend } from './controls';
import PlotCanvas from './PlotCanvas';

const DATASET = 'outliers';
const PLOT_SEED = 7;
/** Starts on the worst case, so the first thing a reader sees is the problem. */
const START_INDEX = 0;

const WIDTH = 320;
const HEIGHT = 150;
const PAD = 22;

const VERDICTS: Record<FitVerdict, { title: string; text: string }> = {
	overfitting: { title: 'Overfitting', text: 'The model memorizes noise. Training looks great, but the score on new data is far lower.' },
	good: { title: 'Good fit', text: 'The model follows the real pattern and ignores most of the noise. Both scores are high and close together.' },
	underfitting: { title: 'Underfitting', text: 'The model is too simple for the pattern. Both scores are low, so more training would not help.' },
};

function FitChart({ curve, index }: { curve: FitPoint[]; index: number }) {
	const x = (i: number) => PAD + (i / (curve.length - 1)) * (WIDTH - 2 * PAD);
	const y = (v: number) => HEIGHT - PAD - ((v - 0.6) / 0.4) * (HEIGHT - 2 * PAD);
	const line = (pick: (p: FitPoint) => number) => curve.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(pick(p)).toFixed(1)}`).join('');
	return (
		<figure class="m-0">
			<figcaption class={fieldLabelClass}>Accuracy at each k, averaged over 20 samples</figcaption>
			<svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} class="mt-1 block w-full" role="img" aria-label="Training accuracy falls steadily as k grows, while test accuracy rises to a peak and then falls.">
				{[0.6, 0.8, 1].map((v) => (
					<g>
						<line x1={PAD} x2={WIDTH - PAD} y1={y(v)} y2={y(v)} stroke="var(--viz-grid)" />
						<text x={PAD - 4} y={y(v) + 3} text-anchor="end" font-size="8" fill="var(--viz-muted-ink)">{v * 100}%</text>
					</g>
				))}
				<line x1={x(index)} x2={x(index)} y1={PAD - 8} y2={HEIGHT - PAD} stroke="var(--viz-ink)" stroke-dasharray="3 3" />
				<path d={line((p) => p.train)} fill="none" stroke="var(--viz-series-1)" stroke-width="2" />
				<path d={line((p) => p.test)} fill="none" stroke="var(--viz-series-2)" stroke-width="2" stroke-dasharray="6 3" />
				<text x={PAD} y={HEIGHT - 6} font-size="8" fill="var(--viz-muted-ink)">k = 1, flexible</text>
				<text x={WIDTH - PAD} y={HEIGHT - 6} text-anchor="end" font-size="8" fill="var(--viz-muted-ink)">k = 85, rigid</text>
			</svg>
			<div class="mt-1 flex gap-4 text-xs text-(--viz-muted-ink)">
				<span class="flex items-center gap-1.5"><svg width="18" height="6" aria-hidden="true"><line x1="0" x2="18" y1="3" y2="3" stroke="var(--viz-series-1)" stroke-width="2" /></svg>Training</span>
				<span class="flex items-center gap-1.5"><svg width="18" height="6" aria-hidden="true"><line x1="0" x2="18" y1="3" y2="3" stroke="var(--viz-series-2)" stroke-width="2" stroke-dasharray="6 3" /></svg>Test</span>
			</div>
		</figure>
	);
}

/** One model, from too flexible to too rigid, with the boundary and both scores side by side. */
export default function FitExplorer() {
	const [index, setIndex] = useState(START_INDEX);
	const curve = useMemo(() => knnFitCurve(DATASET), []);
	const { train, test } = useMemo(() => sampleSplit(DATASET, PLOT_SEED), []);
	const point = curve[index];
	const bestTest = Math.max(...curve.map((p) => p.test));
	const verdict = VERDICTS[fitVerdict(point, bestTest)];
	const scene = useMemo(
		() => withHeldOut({ region: (x, y) => knnClassify(train, point.k, x, y), points: classPoints(train) }, test),
		[point.k],
	);

	return (
		<section class="not-content my-8 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) shadow-sm sm:p-5" aria-label="Overfitting and underfitting explorer">
			<label class="flex flex-col gap-1">
				<span class={`flex justify-between ${fieldLabelClass}`}>
					Neighbors k
					<output class="font-mono tabular-nums text-(--viz-ink)">{point.k}</output>
				</span>
				<input
					type="range"
					class="w-full accent-(--sl-color-accent)"
					min={0}
					max={FIT_NEIGHBORS.length - 1}
					step={1}
					value={index}
					aria-valuetext={`k equals ${point.k}: ${verdict.title}`}
					onInput={(e) => setIndex(Number(e.currentTarget.value))}
				/>
			</label>
			<div aria-live="polite" class="mt-3 min-h-16">
				<p class="font-display m-0 text-lg font-semibold">{verdict.title}</p>
				<p class="m-0 text-sm leading-snug">{verdict.text}</p>
			</div>
			<div class="mt-3 grid gap-4 md:grid-cols-2">
				<div>
					<PlotCanvas scene={scene} class="block w-full rounded-lg border border-(--viz-border)" role="img" aria-label={`KNN decision regions with k equal to ${point.k}.`} />
					<div class="mt-2 text-xs text-(--viz-muted-ink)"><PlotLegend legend={['Class A', 'Class B']} /></div>
					<p class="m-0 mt-1 text-xs leading-relaxed text-(--viz-muted-ink)">One sample. Solid points train the model, and hollow points test it.</p>
				</div>
				<div class="flex flex-col gap-3">
					<FitChart curve={curve} index={index} />
					<MetricList
						class="border-t border-(--viz-border) pt-3"
						metrics={[
							{ label: 'Training accuracy', value: percent(point.train) },
							{ label: 'Test accuracy', value: percent(point.test) },
							{ label: 'Gap', value: `${((point.train - point.test) * 100).toFixed(1)} points` },
						]}
					/>
				</div>
			</div>
		</section>
	);
}
