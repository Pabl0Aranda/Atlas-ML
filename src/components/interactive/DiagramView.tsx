import type { Diagram } from '@/types/viz';

const WIDTH = 640;
const MIN_HEIGHT = 240;
const ROW_HEIGHT = 46;
const PAD = 26;

const seriesColor = (cls?: number) => (cls === undefined ? 'var(--viz-muted)' : `var(--viz-series-${(cls % 4) + 1})`);

/** Renders a tree or a network. Edge width tracks |weight|, edge color its sign. */
export default function DiagramView({ diagram }: { diagram: Diagram }) {
	// Taller columns get more room so stacked node labels never collide.
	const columns = new Map<number, number>();
	for (const n of diagram.nodes) columns.set(n.x, (columns.get(n.x) ?? 0) + 1);
	const isTree = diagram.edges.every((e) => e.weight === undefined);
	const HEIGHT = isTree ? MIN_HEIGHT : Math.max(MIN_HEIGHT, (Math.max(...columns.values()) + 1) * ROW_HEIGHT);
	const px = (x: number) => PAD + x * (WIDTH - 2 * PAD);
	const py = (y: number) => PAD + y * (HEIGHT - 2 * PAD);
	const maxWeight = Math.max(...diagram.edges.map((e) => Math.abs(e.weight ?? 0)), 1e-6);
	const dense = isTree && diagram.nodes.length > 15;

	return (
		<figure class="m-0">
			<figcaption class="text-xs font-medium text-(--viz-muted-ink)">{diagram.title}</figcaption>
			<svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} class="mt-1 block w-full" role="img" aria-label={diagram.title}>
				{diagram.edges.map((e) => {
					const from = diagram.nodes[e.from];
					const to = diagram.nodes[e.to];
					const magnitude = e.weight === undefined ? 0.4 : Math.abs(e.weight) / maxWeight;
					return (
						<line
							x1={px(from.x)}
							y1={py(from.y)}
							x2={px(to.x)}
							y2={py(to.y)}
							stroke={e.weight === undefined ? 'var(--viz-muted)' : seriesColor(e.weight >= 0 ? 1 : 0)}
							stroke-width={0.5 + magnitude * 2.5}
							stroke-opacity={e.weight === undefined ? 1 : 0.25 + magnitude * 0.6}
						/>
					);
				})}
				{diagram.nodes.map((n) => (
					<g transform={`translate(${px(n.x)} ${py(n.y)})`}>
						<circle r={dense ? 7 : 11} fill="var(--viz-surface)" stroke={seriesColor(n.cls)} stroke-width="2" />
						<circle r={dense ? 7 : 11} fill={seriesColor(n.cls)} fill-opacity={n.strength ?? 1} />
						{n.label && !dense && (
							<text y="-16" text-anchor="middle" font-size="12" fill="var(--viz-ink)" class="font-mono">
								{n.label}
							</text>
						)}
					</g>
				))}
			</svg>
		</figure>
	);
}
