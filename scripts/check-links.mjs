// Checks every internal link and asset in the built site, so a broken link fails CI
// before it reaches main. Unlike the source link test, it sees what the build really
// produced: links generated from the catalog, assets, and the subfolder prefix.
//
// Usage: BASE_PATH=/Atlas-ML node scripts/check-links.mjs [dist]

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const ORIGIN = 'http://site.invalid';
const ATTRIBUTE = /\s(?:href|src)="([^"]*)"/g;
const ID = /\sid="([^"]+)"/g;

/** @param {string} dir @returns {string[]} */
const walk = (dir) =>
	readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
		const path = join(dir, entry.name);
		return entry.isDirectory() ? walk(path) : [path];
	});

/** Turns HTML character references, such as &amp; or &#x26;, back into characters. @param {string} text */
const decodeEntities = (text) =>
	text
		.replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
		.replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(Number(dec)))
		.replaceAll('&quot;', '"')
		.replaceAll('&amp;', '&');

/** @param {string} file */
const isFile = (file) => existsSync(file) && statSync(file).isFile();

/**
 * Every internal link in the built site that points at a missing page, file or anchor.
 * @param {string} root The build folder, such as dist.
 * @param {string} [base] The subfolder the site is served from, such as /Atlas-ML.
 * @returns {{ page: string, link: string, reason: string }[]}
 */
export function findBrokenLinks(root, base = '/') {
	const prefix = base.replace(/\/$/, '');
	/** @type {Map<string, Set<string>>} */
	const ids = new Map();
	const idsIn = (/** @type {string} */ file) => {
		if (!ids.has(file)) ids.set(file, new Set([...readFileSync(file, 'utf8').matchAll(ID)].map((m) => m[1])));
		return /** @type {Set<string>} */ (ids.get(file));
	};

	/** @param {string} path A decoded URL path inside the site. */
	const fileFor = (path) => {
		const local = path.slice(prefix.length) || '/';
		const candidates = local.endsWith('/')
			? [join(root, local, 'index.html')]
			: [join(root, local), join(root, local, 'index.html'), join(root, `${local}.html`)];
		return candidates.find(isFile);
	};

	const broken = [];
	for (const file of walk(root).filter((f) => f.endsWith('.html'))) {
		const page = `${prefix}/${relative(root, file).split(sep).join('/')}`.replace(/index\.html$/, '');
		const pageUrl = new URL(page, ORIGIN);
		for (const [, raw] of readFileSync(file, 'utf8').matchAll(ATTRIBUTE)) {
			const link = decodeEntities(raw);
			if (!link || link.startsWith('data:')) continue;
			const url = new URL(link, pageUrl);
			// Other sites, and mail or phone links, are out of scope.
			if (url.origin !== ORIGIN) continue;

			const path = decodeURIComponent(url.pathname);
			if (prefix && path !== prefix && !path.startsWith(`${prefix}/`)) {
				broken.push({ page, link, reason: `outside ${prefix}/` });
				continue;
			}
			const target = path === pageUrl.pathname ? file : fileFor(path);
			if (!target) {
				broken.push({ page, link, reason: 'missing page or file' });
				continue;
			}
			const anchor = decodeURIComponent(url.hash.slice(1));
			if (anchor && target.endsWith('.html') && !idsIn(target).has(anchor)) {
				broken.push({ page, link, reason: 'missing anchor' });
			}
		}
	}
	return broken;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
	const root = process.argv[2] ?? 'dist';
	const base = process.env.BASE_PATH ?? '/';
	const pages = walk(root).filter((f) => f.endsWith('.html')).length;
	const broken = findBrokenLinks(root, base);
	for (const { page, link, reason } of broken) console.error(`${page}: ${link} (${reason})`);
	console.log(`Checked ${pages} pages under ${base}: ${broken.length} broken ${broken.length === 1 ? 'link' : 'links'}.`);
	process.exit(broken.length ? 1 : 0);
}
