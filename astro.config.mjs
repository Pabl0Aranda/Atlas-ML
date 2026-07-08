// @ts-check
import { defineConfig } from 'astro/config';
import preact from '@astrojs/preact';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	integrations: [
		starlight({
			title: 'Atlas ML',
			logo: { src: './src/assets/logo.svg' },
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
				{ label: 'Catalog', link: '/catalog/' },
				{ label: 'Data Preparation', collapsed: true, items: [{ autogenerate: { directory: 'data-preparation' } }] },
				{ label: 'Validation', collapsed: true, items: [{ autogenerate: { directory: 'validation' } }] },
				{ label: 'Testing', collapsed: true, items: [{ autogenerate: { directory: 'testing' } }] },
				{
					label: 'Supervised Learning',
					collapsed: true,
					items: [
						{ slug: 'algorithms/linear-regression' },
						{ slug: 'algorithms/polynomial-regression' },
						{ slug: 'algorithms/decision-tree' },
						{ slug: 'algorithms/knn' },
						{ slug: 'algorithms/svm' },
					],
				},
				{
					label: 'Unsupervised Learning',
					collapsed: true,
					items: [{ slug: 'algorithms/k-means' }, { slug: 'algorithms/pca' }],
				},
				{
					label: 'Statistical Learning',
					collapsed: true,
					items: [
						{ slug: 'algorithms/naive-bayes' },
						{ slug: 'algorithms/logistic-regression' },
						{ slug: 'algorithms/linear-discriminant-analysis' },
						{ slug: 'algorithms/gaussian-mixture-models' },
						{ slug: 'algorithms/ridge-regression' },
						{ slug: 'algorithms/lasso-regression' },
						{ slug: 'algorithms/elastic-net' },
						{ slug: 'algorithms/gaussian-process-regression' },
					],
				},
				{
					label: 'Deep Learning',
					collapsed: true,
					items: [{ slug: 'algorithms/perceptron' }, { slug: 'algorithms/mlp' }],
				},
				{
					label: 'Reinforcement Learning',
					collapsed: true,
					items: [{ slug: 'algorithms/q-learning' }, { slug: 'algorithms/sarsa' }, { slug: 'algorithms/deep-q-networks' }, { slug: 'algorithms/proximal-policy-optimization' }],
				},
				{ label: 'Concepts', collapsed: true, items: [{ autogenerate: { directory: 'concepts' } }] },
			],
		}),
		preact(),
	],
	vite: {
		plugins: [tailwindcss()],
	},
});
