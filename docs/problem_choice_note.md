# Problem-choice note

I chose **Problem 1** (episode-analysis / data-quality tool). It is fully deterministic — no model
API, no inference latency or cost — the safer bet for finishing end-to-end with real evidence
rather than a partially-working pipeline. It also maps onto the case study's strongest lead:
Engineer C's calibration hypothesis in Problem 3. Building the tool first meant Problem 3 could be
grounded in what it actually measures (used-vs-declared calibration mismatch, multi-view
reprojection residual, cross-stream clock offset) rather than written in the abstract, and let me
demonstrate the 3D-geometry/synchronization judgement the grading criteria call out.

Had I taken Problem 2 instead, I would have built the ACCEPT/REVIEW/REJECT label-QC loop using a
constructed human-reviewed-vs-auto-label split (DexYCB ships none natively), scored with
confidence/agreement-based abstention, focused on false-acceptance cost over aggregate accuracy —
the assignment's explicit key requirement.
