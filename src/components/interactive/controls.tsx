import type { Metric, ParamSpec } from '@/types/viz';

export const TICK_MS = 140;
export const SPEEDS = [1, 2, 4];

export const buttonClass =
	'cursor-pointer rounded-md border border-(--viz-border) bg-(--viz-surface) px-3 py-1.5 text-sm font-medium text-(--viz-ink) transition-colors hover:border-(--viz-ink) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--sl-color-accent) disabled:cursor-not-allowed disabled:opacity-45';
export const primaryButtonClass =
	'cursor-pointer rounded-md border border-(--sl-color-accent) bg-(--sl-color-accent) px-3 py-1.5 text-sm font-semibold text-(--on-accent) transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--sl-color-accent) disabled:cursor-not-allowed disabled:opacity-45';
export const selectClass = 'rounded-md border border-(--viz-border) bg-(--viz-surface) px-2 py-1.5 text-sm text-(--viz-ink)';
export const fieldLabelClass = 'text-xs font-medium text-(--viz-muted-ink)';

export function ParamControl({ param, value, onChange }: { param: ParamSpec; value: number; onChange: (v: number) => void }) {
	if (param.options) {
		return (
			<label class="flex flex-col gap-1">
				<span class={fieldLabelClass}>{param.label}</span>
				<select class={selectClass} value={value} onChange={(e) => onChange(Number(e.currentTarget.value))}>
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

export function PlotLegend({ legend }: { legend: string[] }) {
	return (
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
	);
}

export function MetricList({ metrics, class: extra = '' }: { metrics: Metric[]; class?: string }) {
	return (
		<dl class={`m-0 grid grid-cols-2 gap-x-4 gap-y-2 ${extra}`}>
			{metrics.map((m) => (
				<div class="m-0">
					<dt class="text-xs text-(--viz-muted-ink)">{m.label}</dt>
					<dd class="m-0 font-mono text-base tabular-nums">{m.value}</dd>
				</div>
			))}
		</dl>
	);
}
