import { useState } from 'preact/hooks';
import { hamming, jaccardDistance, levenshteinTable } from '@/lib/ml/distance';
import { wordSet } from '@/lib/viz/distanceDemos';
import { fieldLabelClass, MetricList } from './controls';

const MAX_WORD = 16;
const MAX_TEXT = 80;
const frameClass = 'not-content my-6 rounded-2xl border border-(--viz-border) bg-(--viz-panel) p-3 text-(--viz-ink) sm:p-5';
const inputClass = 'w-full rounded-md border border-(--viz-border) bg-(--viz-surface) px-2 py-1.5 font-mono text-sm text-(--viz-ink)';
const cellClass = 'border border-(--viz-border) px-2 py-1 text-center';

function TextField({ label, value, maxLength, onInput }: { label: string; value: string; maxLength: number; onInput: (v: string) => void }) {
	return (
		<label class="flex min-w-0 flex-1 flex-col gap-1">
			<span class={fieldLabelClass}>{label}</span>
			<input type="text" class={inputClass} value={value} maxLength={maxLength} onInput={(e) => onInput(e.currentTarget.value)} />
		</label>
	);
}

/** The table Levenshtein distance fills in. Its bottom-right cell is the answer. */
export function EditDistanceDemo() {
	const [first, setFirst] = useState('kitten');
	const [second, setSecond] = useState('sitting');
	const table = levenshteinTable(first, second);
	const lastRow = first.length;
	const lastColumn = second.length;

	return (
		<section class={frameClass} aria-label="Edit distance demo">
			<div class="flex flex-wrap gap-3">
				<TextField label="First word" value={first} maxLength={MAX_WORD} onInput={setFirst} />
				<TextField label="Second word" value={second} maxLength={MAX_WORD} onInput={setSecond} />
			</div>
			<div class="mt-3 overflow-x-auto">
				<table class="m-0 border-collapse font-mono text-sm tabular-nums">
					<thead>
						<tr>
							<th class={cellClass} scope="col"><span class="sr-only">First word</span></th>
							<th class={cellClass} scope="col"><span class="sr-only">Empty</span></th>
							{[...second].map((c) => <th class={cellClass} scope="col">{c}</th>)}
						</tr>
					</thead>
					<tbody>
						{table.map((row, i) => (
							<tr>
								<th class={cellClass} scope="row">{i === 0 ? <span class="sr-only">Empty</span> : first[i - 1]}</th>
								{row.map((value, j) => (
									<td class={`${cellClass} ${i === lastRow && j === lastColumn ? 'bg-(--sl-color-accent) font-semibold text-(--on-accent)' : ''}`}>{value}</td>
								))}
							</tr>
						))}
					</tbody>
				</table>
			</div>
			<p class="m-0 mt-3 text-sm leading-snug" aria-live="polite">
				Turning “{first}” into “{second}” takes at least <span class="font-mono">{table[lastRow][lastColumn]}</span> single-character edits.
			</p>
		</section>
	);
}

export function SetDistanceDemo() {
	const [first, setFirst] = useState('the cat sat on the mat');
	const [second, setSecond] = useState('the dog sat on the log');
	const words = [wordSet(first), wordSet(second)] as const;
	const shared = [...words[0]].filter((w) => words[1].has(w));
	const positions = hamming(first, second);

	return (
		<section class={frameClass} aria-label="Set distance demo">
			<div class="flex flex-wrap gap-3">
				<TextField label="First text" value={first} maxLength={MAX_TEXT} onInput={setFirst} />
				<TextField label="Second text" value={second} maxLength={MAX_TEXT} onInput={setSecond} />
			</div>
			<MetricList
				class="mt-3 sm:grid-cols-3"
				metrics={[
					{ label: 'Hamming, by character', value: positions === null ? 'lengths differ' : String(positions) },
					{ label: 'Jaccard distance, by word', value: jaccardDistance(words[0], words[1]).toFixed(3) },
					{ label: 'Shared words', value: `${shared.length} of ${new Set([...words[0], ...words[1]]).size}` },
				]}
			/>
		</section>
	);
}
