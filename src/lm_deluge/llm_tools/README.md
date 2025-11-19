# LLM Tools

The "tools" in this repo are not of two kinds:
- (1) Pre-made Tool objects that we consider generally useful and want to make available to LLMs. Examples include the Todo list, the Filesystem, and Subagents. These interoperate with LM Deluge client API so you don't have to define your own tools for these purposes.
- (2) Tools built WITH LLMs. These are like templates or helpful prompts that I found myself reaching for over and over again, that I thought may as well be crystallized into something re-usable. These tools are basically LLM-powered functions/workflows -- they are built out of input/output definitions, a prompt, and response parsing. That's about it!
