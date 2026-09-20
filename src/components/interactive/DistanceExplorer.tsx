import { useMemo, useRef, useState } from 'preact/hooks';
import { generateDataset } from '@/lib/ml/datasets';
import { inverseCovariance2d } from '@/lib/ml/distance';
import { allDistances, EXPLORER_METRICS, explorerScene, measure, type ExplorerContext, type ExplorerMetric } from '@/lib/viz/distanceDemos';
import { moveProbe, probeStep, type Point2D } from '@/lib/viz/interaction';
import { buttonClass, fieldLabelClass, MetricList, ParamControl, PlotLegend, primaryButtonClass, selectClass } from './controls';
import PlotCanvas from './PlotCanvas';

const P_PARAM = { key: 'p', label: 'Exponent p', value: 3, min: 1, max: 8, step: 0.5 };

export default function DistanceExplorer({ initialMetric = 'euclidean' }: { initialMetric?: ExplorerMetric }) {
	const [metric, setMetric] = useState<ExplorerMetric>(initialMetric);
	const [p, setP] = useState(P_PARAM.value);
	const [points, setPoints] = useState<{ a: Point2D; b: Point2D }>({ a: { x: 0.1, y: 0.2 }, b: { x: 0.5, y: -0.1 } });
	const [moving, setMoving] = useState<'a' | 'b'>('b');
	const canvasRef = useRef<HTMLCanvasElement>(null);

	const samples = useMemo(() => generateDataset('ellipse'), []);
	const inverseCovariance = useMemo(() => inverseCovariance2d(samples), [samples]);
	const context: ExplorerContext = { p, inverseCovariance, samples };
	const scene = useMemo(() => explorerScene(metric, points.a, points.b, context), [metric, p, points]);
	const chosen = EXPLORER_METRICS.find((m) => m.value === metric)!;
	const distance = measure(metric, points.a, points.b, context).toFixed(3);

	const place = (at: Point2D) => setPoints((current) => ({ ...current, [moving]: at }));

	const onClick = (event: MouseEvent) => {
		const rect = canvasRef.current!.getBoundingClientRect();
		place({ x: ((event.clientX - rect.left) / rect.width) * 2 - 1, y: 1 - ((event.clientY - rect.top) / rect.height) * 2 });
	};

	const onKey = (event: KeyboardEvent) => {
		const next = moveProbe(points[moving], event.key, probeStep(undefined, event.shiftKey));
		if (!next) return;
		event.preventDefault();
		place(next);
	};

	return (
		<section class="not-content my-8 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) shadow-sm sm:p-5" aria-label="Distance explorer">
			<div class="grid gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
				<div>
					<PlotCanvas
						scene={scene}
						canvasRef={canvasRef}
						class="block w-full cursor-crosshair touch-manipulation rounded-lg border border-(--viz-border) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--sl-color-accent)"
						role="application"
						aria-roledescription="plot"
						aria-label={`Distance plot. Arrow keys move point ${moving.toUpperCase()}.`}
						tabIndex={0}
						onClick={onClick}
						onKeyDown={onKey}
					/>
					<div class="mt-2 min-h-5 text-xs text-(--viz-muted-ink)"><PlotLegend legend={['Point A', 'Point B']} /></div>
				</div>
				<div class="flex flex-col gap-3">
					<label class="flex flex-col gap-1">
						<span class={fieldLabelClass}>Distance</span>
						<select class={selectClass} value={metric} onChange={(e) => setMetric(e.currentTarget.value as ExplorerMetric)}>
							{EXPLORER_METRICS.map((m) => <option value={m.value}>{m.label}</option>)}
						</select>
					</label>
					{metric === 'minkowski' && <ParamControl param={P_PARAM} value={p} onChange={setP} />}
					<div>
						<p class={`m-0 mb-1 ${fieldLabelClass}`}>Click or arrow keys move</p>
						<div class="grid grid-cols-2 gap-2">
							{(['a', 'b'] as const).map((point) => (
								<button
									type="button"
									class={point === moving ? primaryButtonClass : buttonClass}
									aria-pressed={point === moving}
									onClick={() => setMoving(point)}
								>
									Point {point.toUpperCase()}
								</button>
							))}
						</div>
					</div>
					<p class="m-0 text-xs leading-relaxed text-(--viz-muted-ink)">Rings show equal distance from point A. Focus the plot to use the arrow keys, and hold Shift for larger steps.</p>
				</div>
			</div>
			<p class="m-0 mt-3 min-h-10 text-sm leading-snug" aria-live="polite">{chosen.label} distance: <span class="font-mono tabular-nums">{distance}</span>. {chosen.note}</p>
			<MetricList class="mt-3 border-t border-(--viz-border) pt-3 sm:grid-cols-3" metrics={allDistances(points.a, points.b, context)} />
		</section>
	);
}
