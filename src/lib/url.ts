interface HastNode {
	type: string;
	tagName?: string;
	properties?: Record<string, unknown>;
	children?: HastNode[];
}

/** Prefixes a root-relative path with the folder the site is served from. */
export function withBase(path: string, base: string = import.meta.env.BASE_URL): string {
	return `${base.replace(/\/$/, '')}${path}`;
}

const isInternal = (href: unknown): href is string => typeof href === 'string' && href.startsWith('/') && !href.startsWith('//');

/** Lets pages keep plain links like /catalog/, wherever the site is served from. */
export function rehypeBaseLinks({ base = '/' }: { base?: string } = {}) {
	const visit = (node: HastNode) => {
		if (node.tagName === 'a' && node.properties && isInternal(node.properties.href)) {
			node.properties.href = withBase(node.properties.href, base);
		}
		node.children?.forEach(visit);
	};
	return visit;
}
