/**
 * Builds the LaTeX for a sandbox's live equation: the model's formula with its current
 * numbers in place. Everything here is pure, so specs can call it from `equation`.
 */

/** A number for display: fixed digits, no "-0", and a word instead of NaN or infinity. */
export function num(value: number, digits = 2): string {
	if (Number.isNaN(value)) return '\\text{undefined}';
	if (!Number.isFinite(value)) return value > 0 ? '\\infty' : '-\\infty';
	const text = value.toFixed(digits);
	return /^-0\.?0*$/.test(text) ? text.slice(1) : text;
}

/** One term of a sum, with its sign pulled out: "+ 0.71x" or "- 0.30x^{2}". Leading terms keep a bare minus. */
function term(coefficient: number, variable: string, first: boolean, digits: number): string {
	const shown = num(Math.abs(coefficient), digits);
	const sign = coefficient < 0 || Object.is(coefficient, -0) ? '-' : '+';
	// A command such as \infty would run into the variable, so it gets braces.
	const body = shown.startsWith('\\') ? `{${shown}}${variable}` : `${shown}${variable}`;
	if (first) return sign === '-' && Number(shown) !== 0 ? `-${body}` : body;
	return ` ${sign} ${body}`;
}

/**
 * A sum of coefficients times variables, such as "0.12 + 0.71x - 0.30x^{2}". Terms whose
 * coefficient is exactly zero are left out, so lasso's dropped terms disappear.
 */
export function sum(coefficients: number[], variables: string[], digits = 2): string {
	const terms = coefficients
		.map((c, i) => ({ c, v: variables[i] }))
		.filter(({ c }, i) => i === 0 || c !== 0)
		.filter(({ c }, i, all) => !(i === 0 && c === 0 && all.length > 1));
	if (!terms.length) return '0';
	return terms.map(({ c, v }, i) => term(c, v, i === 0, digits)).join('');
}

/** The powers of x for a polynomial: "", "x", "x^{2}", and so on. */
export const powersOf = (variable: string, degree: number): string[] =>
	Array.from({ length: degree + 1 }, (_, j) => (j === 0 ? '' : j === 1 ? variable : `${variable}^{${j}}`));

/** A polynomial prediction, such as "\hat{y} = 0.12 + 0.71x". */
export const polynomial = (weights: number[], digits = 2): string => `\\hat{y} = ${sum(weights, powersOf('x', weights.length - 1), digits)}`;

/** A 2D vector, such as "(0.12,\ -0.40)". */
export const vector = (x: number, y: number, digits = 2): string => `(${num(x, digits)},\\ ${num(y, digits)})`;

/** Several lines, aligned on their first "=" or arrow, for KaTeX display mode. */
export const lines = (...rows: string[]): string =>
	`\\begin{aligned} ${rows.map((row) => row.replace(/=|\\leftarrow|\\iff/, (mark) => `&${mark}`)).join(' \\\\ ')} \\end{aligned}`;
