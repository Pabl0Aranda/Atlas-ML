/** The four workflow stages, in order, each with the page that introduces it. */
export const STAGES = [
	{ id: 'prepare', label: 'Prepare', href: '/data-preparation/overview/' },
	{ id: 'train', label: 'Train', href: '/catalog/' },
	{ id: 'validate', label: 'Validate', href: '/validation/overview/' },
	{ id: 'test', label: 'Test', href: '/testing/overview/' },
] as const;

export type StageId = (typeof STAGES)[number]['id'];

/** Concept pages sit where their idea matters most. */
const CONCEPTS: Record<string, StageId> = {
	'concepts/gradient-descent': 'train',
	'concepts/overfitting-underfitting': 'validate',
	'concepts/bias-variance': 'validate',
};

/** Whole sections that belong to one stage. Distance metrics shape how models train. */
const SECTIONS: Record<string, StageId> = {
	'data-preparation': 'prepare',
	algorithms: 'train',
	'distance-metrics': 'train',
	validation: 'validate',
	testing: 'test',
};

/** The stage a docs page belongs to, from its content id, or undefined for pages outside the workflow. */
export function stageFor(id: string): StageId | undefined {
	const path = id.replace(/\/$/, '');
	return CONCEPTS[path] ?? SECTIONS[path.split('/')[0]];
}
