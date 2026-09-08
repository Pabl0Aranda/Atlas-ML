import { useMemo, useState } from 'preact/hooks';
import { timeSeriesComparison } from '@/lib/viz/distanceDemos';
import { MetricList, ParamControl } from './controls';
import PlotCanvas from './PlotCanvas';

const SHIFT = { key: 'shift', label: 'Time shift', value: 10, min: 0, max: 24, step: 1 };

export default function TimeSeriesDemo() {
	const [shift, setShift] = useState(SHIFT.value);
	const { pointwise, warped, scene } = useMemo(() => timeSeriesComparison(shift), [shift]);
	const summary = `Two signals, ${shift} steps apart. Compared point by point they differ by ${pointwise.toFixed(2)}, and after warping by ${warped.toFixed(2)}.`;

	return (
		<section class="not-content my-6 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) sm:p-5" aria-label="Time series demo">
			<div class="grid gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
				<PlotCanvas scene={scene} class="block w-full rounded-lg border border-(--viz-border)" role="img" aria-label={summary} />
				<div class="flex flex-col gap-3">
					<ParamControl param={SHIFT} value={shift} onChange={setShift} />
					<MetricList
						class="grid-cols-1!"
						metrics={[
							{ label: 'Point by point', value: pointwise.toFixed(2) },
							{ label: 'Dynamic time warping', value: warped.toFixed(2) },
						]}
					/>
					<p class="m-0 text-xs leading-relaxed text-(--viz-muted-ink)">Thin lines show which points the warping paired up.</p>
				</div>
			</div>
		</section>
	);
}
