import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { drawGrid, drawScene, readPlotColors } from '@/lib/viz/canvas';
import { DATASET_LABELS, generateDataset } from '@/lib/ml/datasets';
import { loadSpec } from '@/lib/viz/registry';
import { defaultParams } from '@/lib/viz/scene';
import type { Dataset, DatasetKind, Hyperparameters } from '@/types/ml';
import type { AlgorithmId, AlgorithmSpec, ParamSpec } from '@/types/viz';
import DiagramView from './DiagramView';
import HistoryChart from './HistoryChart';

const TICK_MS = 140;
const buttonClass =
	'cursor-pointer rounded-md border border-(--viz-border) bg-(--viz-surface) px-3 py-1.5 text-sm font-medium text-(--viz-ink) transition-colors hover:border-(--viz-ink) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--sl-color-accent) disabled:cursor-not-allowed disabled:opacity-45';
const primaryButtonClass =
	'cursor-pointer rounded-md border border-(--sl-color-accent) bg-(--sl-color-accent) px-3 py-1.5 text-sm font-semibold text-white transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--sl-color-accent) disabled:cursor-not-allowed disabled:opacity-45';
const SPEEDS = [1, 2, 4];
const fieldLabelClass = 'text-xs font-medium text-(--viz-muted-ink)';

function ParamControl({ param, value, onChange }: { param: ParamSpec; value: number; onChange: (v: number) => void }) {
	if (param.options) {
		return (
			<label class="flex flex-col gap-1">
				<span class={fieldLabelClass}>{param.label}</span>
				<select
					class="rounded-md border border-(--viz-border) bg-(--viz-surface) px-2 py-1.5 text-sm text-(--viz-ink)"
					value={value}
					onChange={(e) => onChange(Number(e.currentTarget.value))}
				>
					{param.options.map((o) => <option value={o.value}>{o.label}</option>)}
				</select>
			</label>
		);
	}
	return (
		<label class="flex flex-col gap-1">
			<span class={`flex justify-between ${fieldLabelClass}`}>
				{param.label}
				<output class="font-mono tabular-nums text-(--viz-ink)">{value}</output>
			</span>
			<input
				type="range"
				class="w-full accent-(--sl-color-accent)"
				min={param.min}
				max={param.max}
				step={param.step}
				value={value}
				onInput={(e) => onChange(Number(e.currentTarget.value))}
			/>
		</label>
	);
}

function Runner({ spec }: { spec: AlgorithmSpec }) {
	const [kind, setKind] = useState<DatasetKind>(spec.datasets[0]);
	const [seed, setSeed] = useState(7);
	const [params, setParams] = useState<Hyperparameters>(() => defaultParams(spec));
	const [data, setData] = useState<Dataset>(() => (spec.datasets.length ? generateDataset(spec.datasets[0]) : []));
	const [state, setState] = useState(() => spec.init(data, params));
	const [playing, setPlaying] = useState(false);
	const [speed, setSpeed] = useState(1);
	const [probe, setProbe] = useState({ x: 0.3, y: 0.3 });
	const [hovering, setHovering] = useState(false);
	const [theme, setTheme] = useState(0);
	const [width, setWidth] = useState(0);
	const canvasRef = useRef<HTMLCanvasElement>(null);

	const done = spec.isDone(state, params);
	const editable = spec.editable !== false;
	const scene = useMemo(() => spec.scene(state, data, params), [state, data, params]);

	const aspect = scene.grid ? scene.grid.rows / scene.grid.cols : 1;

	const restart = (nextData: Dataset, nextParams: Hyperparameters) => {
		setPlaying(false);
		setData(nextData);
		setParams(nextParams);
		setState(spec.init(nextData, nextParams));
	};

	const advance = (current: unknown, count: number) => {
		let next = current;
		for (let i = 0; i < count && !spec.isDone(next, params); i++) next = spec.step(next, data, params);
		return next;
	};

	useEffect(() => {
		if (!playing) return;
		if (done) {
			setPlaying(false);
			return;
		}
		const timer = setInterval(() => setState((s: unknown) => advance(s, spec.speed * speed)), TICK_MS);
		return () => clearInterval(timer);
	}, [playing, done, data, params, speed]);

	useEffect(() => {
		const canvas = canvasRef.current!;
		const resize = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
		resize.observe(canvas);
		const themes = new MutationObserver(() => setTheme((t) => t + 1));
		themes.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
		return () => {
			resize.disconnect();
			themes.disconnect();
		};
	}, []);

	useEffect(() => {
		const canvas = canvasRef.current!;
		if (!width) return;
		const ratio = window.devicePixelRatio || 1;
		canvas.width = Math.round(width * ratio);
		canvas.height = Math.round(width * aspect * ratio);
		const ctx = canvas.getContext('2d')!;
		ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		const colors = readPlotColors(canvas);
		if (scene.grid) drawGrid(ctx, width, scene.grid, colors);
		else drawScene(ctx, width, scene, colors);
	}, [scene, width, theme]);

	const toDomain = (event: MouseEvent) => {
		const rect = canvasRef.current!.getBoundingClientRect();
		return {
			x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
			y: 1 - ((event.clientY - rect.top) / rect.height) * 2,
		};
	};

	const addPoint = (event: MouseEvent) => {
		if (!editable) return;
		const { x, y } = toDomain(event);
		restart([...data, { x, y, label: spec.labeled && event.shiftKey ? 1 : 0 }], params);
	};

	const legend = spec.legend(params);
	const caption = spec.caption(state, data, params);
	const series = spec.series?.(state);
	const diagram = spec.diagram?.(state, probe);
	const prediction = scene.region ? legend[scene.region(probe.x, probe.y)] : null;

	return (
		<section class="not-content my-8 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) shadow-sm sm:p-5" aria-label="Interactive sandbox">
			<div class="grid gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
				<div>
					<canvas
						ref={canvasRef}
						class={`block w-full rounded-lg border border-(--viz-border) ${editable ? 'cursor-crosshair' : ''}`}
						style={{ aspectRatio: String(1 / aspect) }}
						role="img"
						aria-label={caption}
						onClick={addPoint}
						onPointerMove={(e) => {
							setProbe(toDomain(e));
							setHovering(true);
						}}
						onPointerLeave={() => setHovering(false)}
					/>
					<div class="mt-2 flex min-h-5 flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-(--viz-muted-ink)">
						<ul class="m-0 flex list-none flex-wrap gap-3 p-0">
							{legend.map((name, i) => (
								<li class="m-0 flex items-center gap-1.5">
									<svg width="10" height="10" viewBox="-6 -6 12 12" aria-hidden="true">
										{[
											<circle r="5" fill="var(--viz-series-1)" />,
											<rect x="-5" y="-5" width="10" height="10" fill="var(--viz-series-2)" />,
											<path d="M0-6 6 5H-6Z" fill="var(--viz-series-3)" />,
											<path d="M0-6 6 0 0 6-6 0Z" fill="var(--viz-series-4)" />,
										][i % 4]}
									</svg>
									{name}
								</li>
							))}
						</ul>
						{hovering && (
							<span class="font-mono tabular-nums">
								{spec.describe
									? spec.describe(state, probe, params)
									: `(${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})${prediction ? ` → ${prediction}` : ''}`}
							</span>
						)}
					</div>
				</div>

				<div class="flex flex-col gap-3">
					{spec.datasets.length > 0 && <label class="flex flex-col gap-1">
						<span class={fieldLabelClass}>Dataset</span>
						<select
							class="rounded-md border border-(--viz-border) bg-(--viz-surface) px-2 py-1.5 text-sm text-(--viz-ink)"
							value={kind}
							onChange={(e) => {
								const next = e.currentTarget.value as DatasetKind;
								setKind(next);
								restart(generateDataset(next, 120, seed), params);
							}}
						>
							{spec.datasets.map((d) => <option value={d}>{DATASET_LABELS[d]}</option>)}
						</select>
					</label>}
					{spec.params.map((param) => (
						<ParamControl
							param={param}
							value={params[param.key]}
							onChange={(value) => restart(data, { ...params, [param.key]: value })}
						/>
					))}
					<div class="grid grid-cols-2 gap-2">
						<button type="button" class={primaryButtonClass} disabled={done} onClick={() => setPlaying(!playing)}>
							{playing ? 'Pause' : 'Play'}
						</button>
						<button type="button" class={buttonClass} disabled={done || playing} onClick={() => setState(advance(state, 1))}>
							Step
						</button>
						<button type="button" class={done ? primaryButtonClass : buttonClass} onClick={() => restart(data, params)}>Reset</button>
						<button
							type="button"
							class={buttonClass}
							onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])}
						>
							Speed {speed}×
						</button>
						{spec.datasets.length > 0 && <button
							type="button"
							class={`${buttonClass} col-span-2`}
							onClick={() => {
								const next = seed + 1;
								setSeed(next);
								restart(generateDataset(kind, 120, next), params);
							}}
						>
							New data
						</button>}
					</div>
					{editable && (
						<p class="m-0 text-xs leading-relaxed text-(--viz-muted-ink)">
							Click the plot to add a sample{spec.labeled ? ', Shift+click for class B' : ''}.
						</p>
					)}
				</div>
			</div>

			<p class="m-0 mt-3 min-h-10 text-sm leading-snug" aria-live="polite">{caption}</p>

			<dl class="m-0 mt-3 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-(--viz-border) pt-3 sm:grid-cols-4">
				{spec.metrics(state, data, params).map((m) => (
					<div class="m-0">
						<dt class="text-xs text-(--viz-muted-ink)">{m.label}</dt>
						<dd class="m-0 font-mono text-base tabular-nums">{m.value}</dd>
					</div>
				))}
			</dl>

			{(series || diagram) && (
				<div class="mt-3 grid gap-4 border-t border-(--viz-border) pt-3">
					{diagram && <DiagramView diagram={diagram} />}
					{series && <HistoryChart series={series} />}
				</div>
			)}
		</section>
	);
}

export default function Sandbox({ algorithm }: { algorithm: AlgorithmId }) {
	const [spec, setSpec] = useState<AlgorithmSpec | null>(null);
	useEffect(() => {
		loadSpec(algorithm).then(setSpec);
	}, [algorithm]);
	if (!spec) {
		return <div class="not-content my-8 h-[36rem] animate-pulse rounded-2xl border border-(--viz-border) bg-(--viz-panel)" aria-label="Loading sandbox" />;
	}
	return <Runner spec={spec} />;
}
