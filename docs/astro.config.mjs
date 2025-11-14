// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'LM Deluge',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/trytaylor/lm-deluge' }],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'index' },
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quick Start', slug: 'getting-started/quickstart' },
					],
				},
				{
					label: 'Core Concepts',
					items: [
						{ label: 'Conversations & Messages', slug: 'core/conversations' },
						{ label: 'Rate Limiting', slug: 'core/rate-limiting' },
						{ label: 'Caching', slug: 'core/caching' },
					],
				},
				{
					label: 'Features',
					items: [
						{ label: 'Tool Use', slug: 'features/tools' },
						{ label: 'MCP Integration', slug: 'features/mcp' },
						{ label: 'Files & Images', slug: 'features/files-images' },
					],
				},
				{
					label: 'Reference',
					autogenerate: { directory: 'reference' },
				},
			],
		}),
	],
});
