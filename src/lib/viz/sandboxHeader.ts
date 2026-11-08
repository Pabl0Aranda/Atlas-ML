import { CATALOG, pageSlug } from '../../data/catalog';
import { stageFor, type StageId } from '../pipeline';
import type { AlgorithmId } from '../../types/viz';

/** Names for sandboxes that teach a concept rather than an algorithm, so the catalog has no entry for them. */
const CONCEPT_NAMES: Partial<Record<AlgorithmId, string>> = {
	'gradient-descent': 'Gradient Descent',
};

export interface SandboxHeading {
	name: string;
	stage: StageId | undefined;
}

/**
 * What a sandbox's header shows: the catalog name and the workflow stage of the page it
 * sits on. One sandbox can appear on two pages, such as linear regression on its own page
 * and on regression metrics, so the page decides.
 */
export function sandboxHeading(algorithm: AlgorithmId, pathname: string, base = '/'): SandboxHeading {
	const prefix = base.replace(/\/$/, '');
	const id = pathname.slice(pathname.startsWith(prefix) ? prefix.length : 0).replace(/^\/|\/$/g, '');
	const entries = CATALOG.filter((e) => e.page?.id === algorithm);
	const entry = entries.find((e) => pageSlug(e) === id) ?? entries[0];
	return { name: entry?.name ?? CONCEPT_NAMES[algorithm] ?? 'Sandbox', stage: stageFor(id) };
}
