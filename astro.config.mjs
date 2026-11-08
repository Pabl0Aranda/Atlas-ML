// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import preact from '@astrojs/preact';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { familySidebar } from './src/data/catalog.ts';
import { rehypeBaseLinks } from './src/lib/url.ts';

// The sitemap, canonical URLs and social images need an absolute origin, so they
// switch on once the deployment sets SITE_URL.
const site = process.env.SITE_URL;
// The folder the site is served from, such as /Atlas-ML on GitHub Pages.
const base = process.env.BASE_PATH || undefined;
const socialImage = site ? new URL(`${base ?? ''}/og-image.png`, site).href : undefined;

// https://astro.build/config
export default defineConfig({
	site,
	base,
	markdown: {
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex, [rehypeBaseLinks, { base }]],
		}),
	},
	integrations: [
		starlight({
			title: 'Atlas ML',
			logo: { src: './src/assets/logo.svg' },
			// Browsers cache tab icons for days, so a new logo needs a new file name.
			favicon: '/icon.svg',
			head: [
				{ tag: 'link', attrs: { rel: 'icon', type: 'image/png', sizes: '32x32', href: `${base ?? ''}/favicon-32.png` } },
				{ tag: 'link', attrs: { rel: 'apple-touch-icon', href: `${base ?? ''}/apple-touch-icon.png` } },
				...(socialImage
					? /** @type {const} */ ([
							{ tag: 'meta', attrs: { property: 'og:image', content: socialImage } },
							{ tag: 'meta', attrs: { name: 'twitter:image', content: socialImage } },
						])
					: []),
			],
			components: {
				PageTitle: './src/components/overrides/PageTitle.astro',
				SocialIcons: './src/components/overrides/SocialIcons.astro',
			},
			description: 'Find any machine learning algorithm, watch it run, read the math and inspect the code.',
			customCss: [
				'katex/dist/katex.min.css',
				'@fontsource-variable/inter',
				'@fontsource-variable/space-grotesk',
				'@fontsource-variable/jetbrains-mono',
				'./src/styles/global.css',
			],
			sidebar: [
				{
					label: 'Get started',
					items: [{ slug: 'start-here' }, { label: 'Catalog', link: '/catalog/' }, { label: 'Compare', link: '/compare/' }, { slug: 'glossary' }],
				},
				{
					label: 'Workflow',
					items: [
						{ label: 'Data Preparation', collapsed: true, items: [{ autogenerate: { directory: 'data-preparation' } }] },
						{ label: 'Validation', collapsed: true, items: [{ autogenerate: { directory: 'validation' } }] },
						{ label: 'Testing', collapsed: true, items: [{ autogenerate: { directory: 'testing' } }] },
					],
				},
				{
					label: 'Algorithms',
					items: /** @type {const} */ (['Supervised Learning', 'Unsupervised Learning', 'Statistical Learning', 'Deep Learning', 'Reinforcement Learning']).map((family) => ({
						label: family,
						collapsed: true,
						items: familySidebar(family),
					})),
				},
				{
					label: 'Foundations',
					items: [
						{ label: 'Concepts', collapsed: true, items: [{ autogenerate: { directory: 'concepts' } }] },
						{ label: 'Distance Metrics', collapsed: true, items: [{ autogenerate: { directory: 'distance-metrics' } }] },
					],
				},
			],
		}),
		preact(),
	],
	vite: {
		plugins: [tailwindcss()],
	},
});
