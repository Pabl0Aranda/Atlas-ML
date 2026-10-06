// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import preact from '@astrojs/preact';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { familySidebar } from './src/data/catalog.ts';

// The sitemap, canonical URLs and social images need an absolute origin, so they
// switch on once the deployment sets SITE_URL.
const site = process.env.SITE_URL;
const socialImage = site ? new URL('/og-image.png', site).href : undefined;

// https://astro.build/config
export default defineConfig({
	site,
	markdown: {
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
	},
	integrations: [
		starlight({
			title: 'Atlas ML',
			logo: { src: './src/assets/logo.svg' },
			head: socialImage
				? [
						{ tag: 'meta', attrs: { property: 'og:image', content: socialImage } },
						{ tag: 'meta', attrs: { name: 'twitter:image', content: socialImage } },
					]
				: [],
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
					items: [{ slug: 'start-here' }, { label: 'Catalog', link: '/catalog/' }, { label: 'Compare', link: '/compare/' }],
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
