import { useEffect, useId, useMemo, useRef, useState } from 'preact/hooks';
import { DATASET_LABELS, generateDataset } from '@/lib/ml/datasets';
import { moveProbe, probeStep, toolOptions, type PlotTool, type Point2D } from '@/lib/viz/interaction';
import { loadSpec } from '@/lib/viz/registry';
import { defaultParams } from '@/lib/viz/scene';
import type { Dataset, DatasetKind, Hyperparameters } from '@/types/ml';
import type { AlgorithmId, AlgorithmSpec } from '@/types/viz';
import { buttonClass, fieldLabelClass, MetricList, ParamControl, PlotLegend, primaryButtonClass, selectClass, SPEEDS, TICK_MS } from './controls';
import DiagramView from './DiagramView';
import HistoryChart from './HistoryChart';
import PlotCanvas from './PlotCanvas';

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
	/** Keeps the readout and cursor visible after a tap or a key press, when no mouse is hovering. */
	const [pinned, setPinned] = useState(false);
	const [tool, setTool] = useState<PlotTool>(0);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const pointerType = useRef('mouse');
	const toolGroup = useId();

	const done = spec.isDone(state, params);
	const editable = spec.editable !== false;
	const scene = useMemo(() => spec.scene(state, data, params), [state, data, params]);

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

	const toDomain = (event: MouseEvent) => {
		const rect = canvasRef.current!.getBoundingClientRect();
		return {
			x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
			y: 1 - ((event.clientY - rect.top) / rect.height) * 2,
		};
	};

	const applyTool = ({ x, y }: Point2D) => {
		if (!editable || tool === 'inspect') return;
		restart([...data, { x, y, label: tool }], params);
	};

	const onPlotClick = (event: MouseEvent) => {
		const at = toDomain(event);
		setProbe(at);
		if (interactive && pointerType.current !== 'mouse') setPinned(true);
		applyTool(at);
	};

	const onPlotKey = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			setPinned(false);
			return;
		}
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			setPinned(true);
			applyTool(probe);
			return;
		}
		const next = moveProbe(probe, event.key, probeStep(scene.grid, event.shiftKey));
		if (!next) return;
		event.preventDefault();
		setProbe(next);
		setPinned(true);
	};

	const legend = spec.legend(params);
	const caption = spec.caption(state, data, params);
	const series = spec.series?.(state);
	const diagram = spec.diagram?.(state, probe);
	const prediction = scene.region ? legend[scene.region(probe.x, probe.y)] : null;
	const probeable = Boolean(scene.region || spec.describe || diagram);
	const interactive = editable || probeable;
	const readout = spec.describe
		? spec.describe(state, probe, params)
		: `(${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})${prediction ? ` → ${prediction}` : ''}`;

	// Announcing every tick would talk over the animation, so the spoken caption waits for a stop.
	const spokenCaption = useRef(caption);
	if (!playing) spokenCaption.current = caption;

	return (
		<section class="not-content my-8 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) shadow-sm sm:p-5" aria-label="Interactive sandbox">
			<div class="grid gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
				<div>
					<div class="relative">
						<PlotCanvas
							scene={scene}
							canvasRef={canvasRef}
							class={`block w-full touch-manipulation rounded-lg border border-(--viz-border) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--sl-color-accent) ${editable ? 'cursor-crosshair' : ''}`}
							role={interactive ? 'application' : 'img'}
							aria-roledescription={interactive ? 'plot' : undefined}
							aria-label={interactive ? `Interactive plot. Arrow keys move the cursor${editable ? ', Enter applies the selected tool' : ''}.` : caption}
							tabIndex={interactive ? 0 : undefined}
							onClick={onPlotClick}
							onKeyDown={onPlotKey}
							onFocus={(e) => e.currentTarget.matches(':focus-visible') && setPinned(true)}
							onBlur={() => setPinned(false)}
							onPointerDown={(e) => {
								pointerType.current = e.pointerType;
							}}
							onPointerMove={(e) => {
								if (e.pointerType !== 'mouse') return;
								setProbe(toDomain(e));
								setHovering(true);
							}}
							onPointerLeave={() => setHovering(false)}
						/>
						{pinned && (
							<span
								class="pointer-events-none absolute size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-(--viz-ink) shadow-[0_0_0_1px_var(--viz-surface)]"
								style={{ left: `${(probe.x + 1) * 50}%`, top: `${(1 - probe.y) * 50}%` }}
								aria-hidden="true"
							/>
						)}
					</div>
					<div class="mt-2 flex min-h-5 flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-(--viz-muted-ink)">
						<PlotLegend legend={legend} />
						<span class="font-mono tabular-nums" aria-live={pinned ? 'polite' : 'off'}>
							{(hovering || pinned) && readout}
						</span>
					</div>
				</div>

				<div class="flex flex-col gap-3">
					{spec.datasets.length > 0 && <label class="flex flex-col gap-1">
						<span class={fieldLabelClass}>Dataset</span>
						<select
							class={selectClass}
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
					{editable && (
						<fieldset class="m-0 border-0 p-0">
							<legend class={`mb-1 p-0 ${fieldLabelClass}`}>Plot tool</legend>
							<div class="flex flex-wrap gap-1.5">
								{toolOptions(legend, spec.labeled).map((option) => (
									<label class="flex min-h-9 cursor-pointer items-center rounded-md border border-(--viz-border) bg-(--viz-surface) px-2.5 text-sm font-medium text-(--viz-ink) transition-colors hover:border-(--viz-ink) has-checked:border-(--sl-color-accent) has-checked:bg-(--sl-color-accent) has-checked:text-(--on-accent) has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-(--sl-color-accent)">
										<input
											type="radio"
											class="sr-only"
											name={toolGroup}
											checked={tool === option.value}
											onChange={() => setTool(option.value)}
										/>
										{option.label}
									</label>
								))}
							</div>
						</fieldset>
					)}
					{spec.params.map((param) => (
						<ParamControl
							param={param}
							value={params[param.key]}
							onChange={(value) => {
								const next = { ...params, [param.key]: value };
								if (param.live) setParams(next);
								else restart(data, next);
							}}
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
					{interactive && (
						<p class="m-0 text-xs leading-relaxed text-(--viz-muted-ink)">
							{editable
								? 'Click or tap the plot to use the tool. From the keyboard, focus the plot, move with the arrow keys and press Enter.'
								: 'Hover or tap the plot to read a location, or focus it and use the arrow keys.'}
						</p>
					)}
				</div>
			</div>

			<p class="m-0 mt-3 min-h-10 text-sm leading-snug">{caption}</p>
			<p class="sr-only" aria-live="polite">{spokenCaption.current}</p>

			<MetricList class="mt-3 border-t border-(--viz-border) pt-3 sm:grid-cols-4" metrics={spec.metrics(state, data, params)} />

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
