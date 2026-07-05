import { useState } from 'preact/hooks';
import type { Series } from '@/types/viz';

const WIDTH = 320;
const HEIGHT = 96;
const PAD = 6;
const MAX_POINTS = 240;

/** Single-series line of a metric over training steps, with a hover readout. */
export default function HistoryChart({ series }: { series: Series }) {
	const [hover, setHover] = useState<number | null>(null);
	const values = series.values.filter(Number.isFinite);
	if (values.length < 2) {
		return (
			<figure class="m-0">
				<figcaption class="text-xs font-medium text-(--viz-muted-ink)">{series.label}</figcaption>
				<p class="m-0 mt-2 text-xs text-(--viz-muted-ink)">Step the algorithm to start the curve.</p>
			</figure>
		);
	}

	const stride = Math.ceil(values.length / MAX_POINTS);
	const sampled = values.filter((_, i) => i % stride === 0 || i === values.length - 1);
	const min = Math.min(...sampled);
	const range = Math.max(...sampled) - min || 1;
	const x = (i: number) => PAD + (i / (sampled.length - 1)) * (WIDTH - 2 * PAD);
	const y = (v: number) => HEIGHT - PAD - ((v - min) / range) * (HEIGHT - 2 * PAD);
	const path = sampled.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join('');
	const active = hover ?? sampled.length - 1;

	const onMove = (event: PointerEvent) => {
		const rect = (event.currentTarget as SVGElement).getBoundingClientRect();
		const ratio = (event.clientX - rect.left) / rect.width;
		setHover(Math.max(0, Math.min(sampled.length - 1, Math.round(ratio * (sampled.length - 1)))));
	};

	return (
		<figure class="m-0">
			<figcaption class="flex justify-between gap-2 text-xs text-(--viz-muted-ink)">
				<span class="font-medium">{series.label}</span>
				<span class="font-mono tabular-nums text-(--viz-ink)">
					step {Math.min(active * stride, values.length - 1)}: {sampled[active].toFixed(4)}
				</span>
			</figcaption>
			<svg
				viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
				class="mt-1 block h-24 w-full touch-none"
				preserveAspectRatio="none"
				role="img"
				aria-label={`${series.label}, latest value ${values.at(-1)!.toFixed(4)}`}
				onPointerMove={onMove}
				onPointerLeave={() => setHover(null)}
			>
				<line x1={PAD} x2={WIDTH - PAD} y1={HEIGHT - PAD} y2={HEIGHT - PAD} stroke="var(--viz-grid)" />
				<path d={path} fill="none" stroke="var(--viz-series-1)" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round" />
				<line x1={x(active)} x2={x(active)} y1={PAD} y2={HEIGHT - PAD} stroke="var(--viz-muted)" vector-effect="non-scaling-stroke" />
			</svg>
		</figure>
	);
}
