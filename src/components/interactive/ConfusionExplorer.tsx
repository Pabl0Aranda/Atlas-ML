import { useState } from 'preact/hooks';
import { f1, precision, recall, type Confusion } from '@/lib/ml/evaluation';
import { accuracyOf, balancedAccuracy, cohenKappa, expectedCost, matthews, negativePredictiveValue, normalizeMatrix, specificity, type Normalization } from '@/lib/ml/indicators';
import { percent } from '@/lib/viz/scene';
import { buttonClass, fieldLabelClass, MetricList, primaryButtonClass, selectClass } from './controls';

const MAX_COUNT = 1_000_000;
const MAX_COST = 10_000;

const EXAMPLES: { label: string; note: string; matrix: Confusion }[] = [
	{ label: 'Disease screening', note: 'It catches 8 of 10 sick people, but most of its alarms are false, so precision is low despite 95.8% accuracy.', matrix: { tp: 8, fn: 2, fp: 40, tn: 950 } },
	{ label: 'Never predicts disease', note: 'It finds nobody, yet accuracy is 99%. Recall, balanced accuracy and MCC all expose it.', matrix: { tp: 0, fn: 10, fp: 0, tn: 990 } },
	{ label: 'Cautious model', note: 'Accuracy rises to 99%, but it misses half of the sick people. Raise the cost of a miss to see why that matters.', matrix: { tp: 5, fn: 5, fp: 5, tn: 985 } },
	{ label: 'Balanced classes', note: 'With equal classes and equal errors, accuracy, F1 and balanced accuracy all agree at 90%.', matrix: { tp: 450, fn: 50, fp: 50, tn: 450 } },
];

const VIEWS: { label: string; value: Normalization; help: string }[] = [
	{ label: 'Counts', value: 'none', help: 'Raw numbers. Always check these first, because a percentage can hide a tiny sample.' },
	{ label: 'By actual class (rows)', value: 'row', help: 'Each row sums to 100%. The diagonal shows recall and specificity.' },
	{ label: 'By prediction (columns)', value: 'column', help: 'Each column sums to 100%. The diagonal shows precision and negative predictive value.' },
	{ label: 'Share of all samples', value: 'all', help: 'All four cells sum to 100%. The diagonal adds up to accuracy.' },
];

const CELLS: { key: keyof Confusion; name: string }[][] = [
	[{ key: 'tp', name: 'True positive' }, { key: 'fn', name: 'False negative' }],
	[{ key: 'fp', name: 'False positive' }, { key: 'tn', name: 'True negative' }],
];

const whole = (raw: string, max: number) => Math.min(max, Math.max(0, Math.floor(Number(raw)) || 0));

/** An editable binary confusion matrix, with every indicator recalculated live. */
export default function ConfusionExplorer() {
	const [matrix, setMatrix] = useState<Confusion>(EXAMPLES[0].matrix);
	const [example, setExample] = useState<number | null>(0);
	const [view, setView] = useState<Normalization>('none');
	const [costs, setCosts] = useState({ fn: 50, fp: 1 });

	const table = normalizeMatrix([[matrix.tp, matrix.fn], [matrix.fp, matrix.tn]], view);
	const shares = normalizeMatrix(table, view === 'none' ? 'all' : 'none');
	const help = VIEWS.find((v) => v.value === view)!.help;

	return (
		<section class="not-content my-8 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) shadow-sm sm:p-5" aria-label="Confusion matrix explorer">
			<p class={`m-0 ${fieldLabelClass}`}>Try an example</p>
			<div class="mt-1.5 flex flex-wrap gap-1.5">
				{EXAMPLES.map((e, i) => (
					<button
						type="button"
						class={i === example ? primaryButtonClass : buttonClass}
						aria-pressed={i === example}
						onClick={() => {
							setMatrix(e.matrix);
							setExample(i);
						}}
					>
						{e.label}
					</button>
				))}
			</div>
			<p class="m-0 mt-2 min-h-10 text-sm leading-snug">{example === null ? 'Your own matrix. Edit any cell to see every indicator respond.' : EXAMPLES[example].note}</p>

			<div class="mt-3 grid gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
				<table class="m-0 w-full table-fixed border-separate border-spacing-1 text-sm">
					<thead>
						<tr>
							<td class="w-20" />
							<th scope="col" class={fieldLabelClass}>Predicted positive</th>
							<th scope="col" class={fieldLabelClass}>Predicted negative</th>
						</tr>
					</thead>
					<tbody>
						{CELLS.map((row, r) => (
							<tr>
								<th scope="row" class={`text-left ${fieldLabelClass}`}>Actually {r === 0 ? 'positive' : 'negative'}</th>
								{row.map((cell, c) => (
									<td class="rounded-lg border border-(--viz-border) p-2 text-center" style={{ background: `color-mix(in srgb, var(--viz-series-1) ${Math.round(shares[r][c] * 45)}%, transparent)` }}>
										<label class="flex flex-col items-center gap-1">
											<span class="text-xs text-(--viz-muted-ink)">{cell.name}</span>
											<input
												type="number"
												min={0}
												max={MAX_COUNT}
												class="w-full max-w-24 rounded-md border border-(--viz-border) bg-(--viz-surface) px-1 py-1 text-center font-mono text-sm text-(--viz-ink)"
												value={matrix[cell.key]}
												onInput={(e) => {
													const value = whole(e.currentTarget.value, MAX_COUNT);
													setMatrix((m) => ({ ...m, [cell.key]: value }));
													setExample(null);
												}}
											/>
										</label>
										{view !== 'none' && <span class="mt-1 block font-mono text-sm tabular-nums">{percent(table[r][c])}</span>}
									</td>
								))}
							</tr>
						))}
					</tbody>
				</table>

				<div class="flex flex-col gap-3">
					<label class="flex flex-col gap-1">
						<span class={fieldLabelClass}>View</span>
						<select class={selectClass} value={view} onChange={(e) => setView(e.currentTarget.value as Normalization)}>
							{VIEWS.map((v) => <option value={v.value}>{v.label}</option>)}
						</select>
					</label>
					<p class="m-0 text-xs leading-relaxed text-(--viz-muted-ink)">{help}</p>
					<div class="grid grid-cols-2 gap-2">
						{(['fn', 'fp'] as const).map((key) => (
							<label class="flex flex-col gap-1">
								<span class={fieldLabelClass}>Cost of a {key === 'fn' ? 'miss' : 'false alarm'}</span>
								<input
									type="number"
									min={0}
									max={MAX_COST}
									class={`w-full ${selectClass}`}
									value={costs[key]}
									onInput={(e) => {
										const value = whole(e.currentTarget.value, MAX_COST);
										setCosts((c) => ({ ...c, [key]: value }));
									}}
								/>
							</label>
						))}
					</div>
				</div>
			</div>

			<MetricList
				class="mt-3 border-t border-(--viz-border) pt-3 sm:grid-cols-4"
				metrics={[
					{ label: 'Accuracy', value: percent(accuracyOf(matrix)) },
					{ label: 'Balanced accuracy', value: percent(balancedAccuracy(matrix)) },
					{ label: 'Precision', value: percent(precision(matrix)) },
					{ label: 'Recall', value: percent(recall(matrix)) },
					{ label: 'Specificity', value: percent(specificity(matrix)) },
					{ label: 'Negative predictive value', value: percent(negativePredictiveValue(matrix)) },
					{ label: 'F1', value: f1(matrix).toFixed(3) },
					{ label: 'MCC', value: matthews(matrix).toFixed(3) },
					{ label: "Cohen's kappa", value: cohenKappa(matrix).toFixed(3) },
					{ label: 'Total error cost', value: String(expectedCost(matrix, costs)) },
				]}
			/>
		</section>
	);
}
