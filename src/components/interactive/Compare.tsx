import { useEffect, useMemo, useState } from 'preact/hooks';
import { compareData, decodeCompare, decodePair, encodeCompare, partnerFor, presetQuery, sharedDatasets, type CompareOption, type ComparePair, type ComparePreset, type CompareSetup } from '@/lib/viz/compareSetup';
import { DATASET_LABELS } from '@/lib/ml/datasets';
import { COMPARE_PRESETS } from '@/lib/viz/presets';
import { loadSpec } from '@/lib/viz/registry';
import { heldOutMetrics, withHeldOut } from '@/lib/viz/sandboxSetup';
import { createTimeline, currentFrame, pushFrame, seekFrame, type Timeline } from '@/lib/viz/timeline';
import type { Dataset, DatasetKind, Hyperparameters } from '@/types/ml';
import type { AlgorithmId, AlgorithmSpec } from '@/types/viz';
import { buttonClass, CopyLinkButton, fieldLabelClass, MetricList, ParamControl, PlotLegend, primaryButtonClass, selectClass, SPEEDS, TICK_MS, TimelineControl } from './controls';
import HistoryChart from './HistoryChart';
import PlotCanvas from './PlotCanvas';

type Frame = [unknown, unknown];

interface PanelProps {
	title: string;
	spec: AlgorithmSpec;
	state: unknown;
	train: Dataset;
	test: Dataset;
	params: Hyperparameters;
	selected: AlgorithmId;
	choices: CompareOption[];
	/** Algorithms left out of the list, because they share no data with the other side. */
	isDisabled: (option: CompareOption) => boolean;
	onPick: (id: AlgorithmId) => void;
	onParam: (key: string, value: number, live: boolean) => void;
}

function Panel({ title, spec, state, train, test, params, selected, choices, isDisabled, onPick, onParam }: PanelProps) {
	const modelScene = useMemo(() => spec.scene(state, train, params), [spec, state, train, params]);
	const scene = useMemo(() => withHeldOut(modelScene, test), [modelScene, test]);
	const caption = spec.caption(state, train, params);
	const series = spec.series?.(state);

	return (
		<section class="flex min-w-0 flex-col gap-3 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 sm:p-4" aria-label={title}>
			<label class="flex flex-col gap-1">
				<span class={fieldLabelClass}>{title}</span>
				<select class={selectClass} value={selected} onChange={(e) => onPick(e.currentTarget.value as AlgorithmId)}>
					{choices.map((o) => <option value={o.id} disabled={isDisabled(o)}>{o.name}</option>)}
				</select>
			</label>
			<PlotCanvas scene={scene} class="block w-full rounded-lg border border-(--viz-border)" role="img" aria-label={caption} />
			<div class="min-h-5 text-xs text-(--viz-muted-ink)"><PlotLegend legend={spec.legend(params)} /></div>
			<p class="m-0 min-h-16 text-sm leading-snug">{caption}</p>
			<MetricList class="border-t border-(--viz-border) pt-3" metrics={[...spec.metrics(state, train, params), ...heldOutMetrics(modelScene.region, test)]} />
			{spec.params.length > 0 && (
				<div class="flex flex-col gap-3 border-t border-(--viz-border) pt-3">
					{spec.params.map((param) => (
						<ParamControl param={param} value={params[param.key]} onChange={(value) => onParam(param.key, value, Boolean(param.live))} />
					))}
				</div>
			)}
			{series && <div class="border-t border-(--viz-border) pt-3"><HistoryChart series={series} /></div>}
		</section>
	);
}

interface ArenaProps {
	pair: ComparePair;
	specA: AlgorithmSpec;
	specB: AlgorithmSpec;
	search: string;
	options: CompareOption[];
	onPair: (pair: ComparePair, search: string) => void;
	onEdit: () => void;
}

/** Runs both algorithms in lockstep. One timeline holds both states, so they rewind together. */
function Arena({ pair, specA, specB, search, options, onPair, onEdit }: ArenaProps) {
	const [setup, setSetup] = useState<CompareSetup>(() => decodeCompare(search, specA, specB));
	const { train, test } = useMemo(() => compareData(setup), [setup.dataset, setup.seed, setup.holdout]);
	const start = (next: CompareSetup): Timeline<Frame> => {
		const data = compareData(next).train;
		return createTimeline<Frame>([specA.init(data, next.paramsA), specB.init(data, next.paramsB)]);
	};
	const [timeline, setTimeline] = useState(() => start(setup));
	const [playing, setPlaying] = useState(false);
	const [speed, setSpeed] = useState(1);

	const [stateA, stateB] = currentFrame(timeline);
	const done = specA.isDone(stateA, setup.paramsA) && specB.isDone(stateB, setup.paramsB);

	const restart = (next: CompareSetup) => {
		setPlaying(false);
		setSetup(next);
		setTimeline(start(next));
		onEdit();
	};

	const seek = (index: number) => {
		setPlaying(false);
		setTimeline((t) => seekFrame(t, index));
	};

	const run = (spec: AlgorithmSpec, state: unknown, params: Hyperparameters, count: number) => {
		let next = spec.snapshot ? spec.snapshot(state) : state;
		for (let i = 0; i < count && !spec.isDone(next, params); i++) next = spec.step(next, train, params);
		return next;
	};

	/** A finished side keeps its last state, so frames stay aligned. */
	const advance = (t: Timeline<Frame>, multiplier: (spec: AlgorithmSpec) => number) => {
		const [a, b] = currentFrame(t);
		return pushFrame<Frame>(t, [run(specA, a, setup.paramsA, multiplier(specA)), run(specB, b, setup.paramsB, multiplier(specB))]);
	};

	useEffect(() => {
		if (!playing) return;
		if (done) {
			setPlaying(false);
			return;
		}
		const timer = setInterval(() => setTimeline((t) => advance(t, (spec) => spec.speed * speed)), TICK_MS);
		return () => clearInterval(timer);
	}, [playing, done, train, setup, speed]);

	const query = () => encodeCompare(pair, setup, specA, specB);
	const changeParam = (side: 'paramsA' | 'paramsB') => (key: string, value: number, live: boolean) => {
		const next = { ...setup, [side]: { ...setup[side], [key]: value } };
		if (!live) restart(next);
		else {
			setSetup(next);
			onEdit();
		}
	};

	const optionOf = (id: AlgorithmId) => options.find((o) => o.id === id)!;
	const canHoldOut = Boolean(specA.holdout && specB.holdout);
	const shared = sharedDatasets(specA, specB);

	return (
		<>
			<div class="mt-4 flex flex-col gap-3 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) sm:p-4 md:sticky md:top-(--sl-nav-height) md:z-5">
				<div class="flex flex-wrap items-end gap-2">
					<label class="flex min-w-36 flex-col gap-1">
						<span class={fieldLabelClass}>Shared dataset</span>
						<select class={selectClass} value={setup.dataset} onChange={(e) => restart({ ...setup, dataset: e.currentTarget.value as DatasetKind })}>
							{shared.map((d) => <option value={d}>{DATASET_LABELS[d]}</option>)}
						</select>
					</label>
					<button type="button" class={primaryButtonClass} disabled={done} onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
					<button type="button" class={buttonClass} disabled={done || playing} onClick={() => setTimeline((t) => advance(t, () => 1))}>Step</button>
					<button type="button" class={buttonClass} disabled={timeline.index === 0} onClick={() => seek(timeline.index - 1)}>Back</button>
					<button type="button" class={buttonClass} onClick={() => restart(setup)}>Reset</button>
					<button type="button" class={buttonClass} onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])}>Speed {speed}×</button>
					<button type="button" class={buttonClass} onClick={() => restart({ ...setup, seed: setup.seed + 1 })}>New data</button>
					<CopyLinkButton label="Copy link" query={query} />
				</div>
				<div class="flex flex-wrap items-center gap-x-6 gap-y-2">
					<div class="min-w-48 flex-1"><TimelineControl index={timeline.index} length={timeline.frames.length} onSeek={seek} /></div>
					{canHoldOut && (
						<label class="flex cursor-pointer items-center gap-2 text-sm">
							<input type="checkbox" class="accent-(--sl-color-accent)" checked={setup.holdout} onChange={(e) => restart({ ...setup, holdout: e.currentTarget.checked })} />
							Hold out 25% for testing
						</label>
					)}
				</div>
				{test.length > 0 && <p class="m-0 text-xs text-(--viz-muted-ink)">Hollow points are held out: neither model trains on them.</p>}
			</div>

			<div class="mt-4 grid gap-4 text-(--viz-ink) md:grid-cols-2">
				<Panel
					title="Algorithm A"
					spec={specA}
					state={stateA}
					train={train}
					test={test}
					params={setup.paramsA}
					selected={pair.a}
					choices={options}
					isDisabled={() => false}
					onPick={(a) => onPair({ a, b: partnerFor(options, a, pair.b) }, query())}
					onParam={changeParam('paramsA')}
				/>
				<Panel
					title="Algorithm B"
					spec={specB}
					state={stateB}
					train={train}
					test={test}
					params={setup.paramsB}
					selected={pair.b}
					choices={options}
					isDisabled={(o) => sharedDatasets(optionOf(pair.a), o).length === 0}
					onPick={(b) => onPair({ a: pair.a, b }, query())}
					onParam={changeParam('paramsB')}
				/>
			</div>
		</>
	);
}

export default function Compare({ options }: { options: CompareOption[] }) {
	const [opened, setOpened] = useState(() => {
		const search = typeof window === 'undefined' ? '' : window.location.search;
		return { pair: decodePair(search, options), search, version: 0 };
	});
	const [specs, setSpecs] = useState<{ a: AlgorithmSpec; b: AlgorithmSpec; version: number } | null>(null);
	const [preset, setPreset] = useState<ComparePreset | null>(null);
	const { pair, search, version } = opened;

	useEffect(() => {
		let current = true;
		Promise.all([loadSpec(pair.a), loadSpec(pair.b)]).then(([a, b]) => {
			if (current) setSpecs({ a, b, version });
		});
		return () => {
			current = false;
		};
	}, [version]);

	const open = (nextPair: ComparePair, nextSearch: string) => setOpened({ pair: nextPair, search: nextSearch, version: version + 1 });

	return (
		<div class="not-content">
			<p class={`m-0 ${fieldLabelClass}`}>Try a pairing</p>
			<div class="mt-1.5 flex flex-wrap gap-1.5">
				{COMPARE_PRESETS.map((p) => (
					<button
						type="button"
						class={p === preset ? primaryButtonClass : buttonClass}
						aria-pressed={p === preset}
						onClick={() => {
							setPreset(p);
							open({ a: p.a, b: p.b }, presetQuery(p));
						}}
					>
						{p.label}
					</button>
				))}
			</div>
			{preset && <p class="m-0 mt-2 text-sm leading-snug text-(--viz-ink)">{preset.note}</p>}

			{specs?.version === version ? (
				<Arena
					key={version}
					pair={pair}
					specA={specs.a}
					specB={specs.b}
					search={search}
					options={options}
					onPair={(next, query) => {
						setPreset(null);
						open(next, query);
					}}
					onEdit={() => setPreset(null)}
				/>
			) : (
				<div class="mt-4 h-[36rem] animate-pulse rounded-2xl border border-(--viz-border) bg-(--viz-panel)" aria-label="Loading comparison" />
			)}
		</div>
	);
}
