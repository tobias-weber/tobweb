import { defineCollection, z } from 'astro:content';

import { glob } from 'astro/loaders';

const projects = defineCollection({
    loader: glob({ base: './src/content/projects', pattern: ['**/*.md', '**/*.mdx'] }),
    schema: ({ image }) => z.object({
        title: z.string(),
        description: z.string(),
        pubDate: z.date(),
        image: z.object({
            src: image(),
            alt: z.string(),
        }),
        tools: z.array(z.string()).optional(),
    }),
});

export const collections = { projects };