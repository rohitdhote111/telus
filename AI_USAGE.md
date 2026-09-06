# AI_USAGE.md

**Tools:**

- **ChatGPT (OpenAI)** — used earlier and separately, for initial brainstorming on how to approach
the assignment and to build my own understanding of the problem space (robot-learning data
pipelines, VLA failure modes) before any implementation started.
- **Claude Code (Anthropic)** — used for dataset investigation, architecture, source code, and
tests for Problem 1, and for drafting Problem 3's reasoning and the summary docs. Disclosed for
the whole submission rather than line-by-line, since involvement was continuous, not pasted
snippets.

## Decisions: options the AI presented, choice made based on my own judgment

- **Dataset sourcing** — Claude checked real download access for DexYCB, HOI4D, HO-3D, Ego4D, and
EPIC-KITCHENS, and proposed a hybrid: real DexYCB calibration data plus a schema-faithful
synthetic episode layer, since the full raw capture (12–119GB) wasn't practical to pull in. I
approved this over a fully synthetic dataset or attempting the full download, since it kept the
geometry/calibration checks grounded in genuine sensor numbers.
- **Threshold values** — Claude proposed initial z-score bands, then several revisions after
validation runs surfaced false positives. I required each revision to be justified by the
validation numbers (precision/recall, false-flag rate) before accepting it, not just an assertion
that it was fixed.



## What I verified rather than took on faith

- Texploring and understanding dataset with basic experiements on it.
- The validation results (100% fault recall, low false-flag rate) before accepting the tool as
done — one AI-added check (intrinsics plausibility) turned out to produce a 100% false-flag rate
on real data and was removed rather than kept.
- The Problem 3 statistical claims (trial-count math, cited drift figures) against the underlying
calculation and this submission's own measured numbers, not repeated as given.

