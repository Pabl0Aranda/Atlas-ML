import { existsSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { GLOSSARY, GLOSSARY_ENTRIES, type GlossaryEntry } from './glossary';

const entries = Object.entries(GLOSSARY) as [string, GlossaryEntry][];

describe('glossary', () => {
	it('keeps definitions short enough for a tooltip', () => {
		const long = entries.filter(([, e]) => e.definition.length > 160).map(([id]) => id);
		expect(long).toEqual([]);
	});

	it('gives every term at least one form to match, in lower case', () => {
		for (const [id, e] of entries) {
			expect(e.forms.length, id).toBeGreaterThan(0);
			for (const form of e.forms) expect(form, id).toBe(form.toLowerCase());
		}
	});

	it('never claims the same form for two terms', () => {
		const forms = entries.flatMap(([, e]) => e.forms);
		expect(forms.filter((f, i) => forms.indexOf(f) !== i)).toEqual([]);
	});

	it('points every "learn more" link at a page that exists', () => {
		const missing = entries
			.filter(([, e]) => e.see && !existsSync(`src/content/docs/${e.see.replace(/^\/|\/$/g, '')}.mdx`))
			.map(([id]) => id);
		expect(missing).toEqual([]);
	});

	it('lists the terms alphabetically', () => {
		const terms = GLOSSARY_ENTRIES.map(([, e]) => e.term);
		expect(terms).toEqual([...terms].sort((a, b) => a.localeCompare(b)));
	});
});
