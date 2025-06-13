// @ts-check
import {defineConfig} from 'astro/config';

import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';
import icon from 'astro-icon';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
    vite: {
        plugins: [tailwindcss()]
    },
    integrations: [
        mdx(),
        icon({
            iconDir: 'src/assets/icons',
            svgoOptions: {
                multipass: true,
                plugins: [
                    {
                        name: 'preset-default',
                        params: {
                            overrides: {
                                "cleanupIds": false // required for gradients to work correctly
                            }
                        }
                    }
                ]
            }
        }),
        sitemap(),
    ],
    site: 'https://tobweb.ch',
});