import { useState } from 'preact/hooks';
import type { Calculations } from '@/types/viz';

/** Open by default where there is room for the table, folded on phones. */
const WIDE = '(min-width: 48rem)';

/** The arithmetic behind the current step, as a small table under the metrics. */
export default function CalculationsPanel({ calculations }: { calculations: Calculations }) {
	const [open] = useState(() => window.matchMedia(WIDE).matches);
	const { title, columns, rows, highlight, note } = calculations;
	return (
		<details class="calculations" open={open}>
			<summary>Show the calculations</summary>
			<div class="calculations-scroll">
				<table>
					<caption>{title}</caption>
					<thead>
						<tr>
							{columns.map((column) => (
								<th scope="col">{column}</th>
							))}
						</tr>
					</thead>
					<tbody>
						{rows.map((row, i) => (
							<tr class={i === highlight ? 'is-highlighted' : undefined}>
								{row.map((cell, j) => (j === 0 ? <th scope="row">{cell}</th> : <td>{cell}</td>))}
							</tr>
						))}
					</tbody>
				</table>
			</div>
			{note && <p class="calculations-note">{note}</p>}
		</details>
	);
}
