import { useMemo, useState } from 'preact/hooks';
import { klDivergence, wasserstein } from '@/lib/ml/distance';
import { distributionBins, distributionScene } from '@/lib/viz/distanceDemos';
import { MetricList, ParamControl } from './controls';
import PlotCanvas from './PlotCanvas';

const mean = (label: string, value: number) => ({ key: label, label, value, min: -0.8, max: 0.8, step: 0.05 });
const spread = (label: string, value: number) => ({ key: label, label, value, min: 0.05, max: 0.4, step: 0.01 });

const CONTROLS = [mean('Mean of P', -0.3), spread('Spread of P', 0.12), mean('Mean of Q', 0.2), spread('Spread of Q', 0.25)];

export default function DistributionDemo() {
	const [values, setValues] = useState(() => CONTROLS.map((c) => c.value));
	const [meanP, spreadP, meanQ, spreadQ] = values;
	const p = useMemo(() => distributionBins(meanP, spreadP), [meanP, spreadP]);
	const q = useMemo(() => distributionBins(meanQ, spreadQ), [meanQ, spreadQ]);
	const scene = useMemo(() => distributionScene(p.values, q.values), [p, q]);
	const forward = klDivergence(p.values, q.values);
	const backward = klDivergence(q.values, p.values);
	const moved = wasserstein(p.values, q.values, p.width);

	return (
		<section class="not-content my-6 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) sm:p-5" aria-label="Distribution demo">
			<div class="grid gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
				<div>
					<PlotCanvas scene={scene} class="block w-full rounded-lg border border-(--viz-border)" role="img" aria-label={`Two distributions. Wasserstein distance ${moved.toFixed(3)}.`} />
					<p class="sr-only" aria-live="polite">{`Wasserstein distance ${moved.toFixed(3)}.`}</p>
					<p class="m-0 mt-2 text-xs text-(--viz-muted-ink)">Solid curve: P. Dashed curve: Q.</p>
				</div>
				<div class="flex flex-col gap-3">
					{CONTROLS.map((control, i) => (
						<ParamControl param={control} value={values[i]} onChange={(v) => setValues((current) => current.map((old, j) => (j === i ? v : old)))} />
					))}
				</div>
			</div>
			<MetricList
				class="mt-3 border-t border-(--viz-border) pt-3 sm:grid-cols-3"
				metrics={[
					{ label: 'KL(P ‖ Q)', value: forward.toFixed(3) },
					{ label: 'KL(Q ‖ P)', value: backward.toFixed(3) },
					{ label: 'Wasserstein', value: moved.toFixed(3) },
				]}
			/>
		</section>
	);
}
