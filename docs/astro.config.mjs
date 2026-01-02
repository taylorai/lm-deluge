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
        { label: "Introduction", link: "/" },
        { label: "Installation", link: "/getting-started/installation/" },
        { label: "Quick Start", link: "/getting-started/quickstart/" },
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
            { label: "Anthropic Skills", slug: "features/skills" },
            { label: "Structured Outputs", slug: "features/structured-outputs" },
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
          label: "Server",
          items: [{ label: "Proxy Server", slug: "server/proxy" }],
        },
        { label: "API Reference", link: "/reference/api/" },
        { label: "Changelog", link: "/changelog/" },
        { label: "Blog", link: "/blog" },
      ],
    }),
  ],
});
