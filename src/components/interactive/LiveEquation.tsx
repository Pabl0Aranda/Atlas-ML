import { useEffect, useMemo, useState } from 'preact/hooks';

type Katex = typeof import('katex').default;

/** KaTeX loads once, and only on pages whose sandbox has an equation. */
let loading: Promise<Katex> | undefined;
const loadKatex = () => (loading ??= import('katex').then((m) => m.default));

/**
 * The model's formula with its current numbers, rendered with KaTeX. It is not a live
 * region: during Play it changes several times a second, which would flood a screen
 * reader. KaTeX's MathML output lets screen readers read it on demand instead.
 */
export default function LiveEquation({ latex }: { latex: string }) {
	const [katex, setKatex] = useState<Katex | null>(null);
	useEffect(() => {
		loadKatex().then(setKatex);
	}, []);
	const html = useMemo(() => katex?.renderToString(latex, { displayMode: true, throwOnError: false }), [katex, latex]);

	return (
		<figure class="live-equation">
			<figcaption class="live-equation-label">Live equation</figcaption>
			{/* KaTeX escapes everything it renders, and the LaTeX comes from the spec, never from visitors. */}
			{html ? <div class="live-equation-math" dangerouslySetInnerHTML={{ __html: html }} /> : <div class="live-equation-math" aria-hidden="true" />}
		</figure>
	);
}
