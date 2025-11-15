// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  integrations: [
    starlight({
      title: "LM Deluge",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/taylorai/lm-deluge",
        },
      ],
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Introduction", slug: "index" },
            { label: "Installation", slug: "getting-started/installation" },
            { label: "Quick Start", slug: "getting-started/quickstart" },
          ],
        },
        {
          label: "Configuring the Client",
          items: [
            { label: "Client Basics", slug: "core/configuring-client" },
            { label: "Rate Limiting", slug: "core/rate-limiting" },
          ],
        },
        {
          label: "Conversations",
          items: [
            { label: "Conversation Builder", slug: "core/conversations/index" },
            { label: "Working with Images", slug: "core/conversations/images" },
            { label: "Working with Files", slug: "core/conversations/files" },
          ],
        },
        {
          label: "Tooling & MCP",
          items: [
            { label: "Tool Use", slug: "features/tools" },
            { label: "MCP Integration", slug: "features/mcp" },
            { label: "Advanced Workflows", slug: "guides/advanced-usage" },
          ],
        },
        {
          label: "Caching & Reliability",
          items: [{ label: "Local & Provider Caching", slug: "core/caching" }],
        },
        {
          label: "Models",
          items: [
            { label: "Supported Providers", slug: "reference/providers" },
            { label: "Using Custom Models", slug: "reference/custom-models" },
          ],
        },
        {
          label: "Reference",
          items: [{ label: "API Reference", slug: "reference/api" }],
        },
        {
          label: "Project Updates",
          items: [{ label: "Changelog", slug: "changelog" }],
        },
      ],
    }),
  ],
});
