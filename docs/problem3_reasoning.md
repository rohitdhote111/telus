# Problem 3 — The VLA Got Worse. What Do You Do Monday Morning?

**Given:** task success ~68% → ~41% after ingesting the latest demonstration batch; offline
metrics changed only moderately. 48 hours, limited compute, limited human review.

**The organizing fact:** offline metrics moved a little, robot success collapsed. That signature is
diagnostic. A model that is under-trained or under-sized degrades offline and online *together*; a
model trained on data that is *self-consistent but systematically wrong* (mis-attached calibration,
a shifted action–observation alignment, drifted normalization statistics) fits that data to a good
offline score and then acts on the real robot in a coordinate system that no longer matches
reality. The investigation below is built around discriminating within that second family — and
around one control that everyone forgets.

## 1. Hypotheses (ranked)

**H1 — The evaluation side changed, not the model.** Robot calibration drift, gripper wear, a
moved camera, a changed eval protocol or task set. Ranked first not because it is most likely, but
because it is the cheapest to rule out and everything else is wasted effort if the 41% is an eval
artifact. *Confidence up:* the old 68% checkpoint, re-run today, also scores ~41%. *Confidence
down:* it still scores ~68%.

**H2 — A systematic pipeline defect in the new batch** (Engineer C's hypothesis, widened beyond
"calibration"): a wrong or stale extrinsics file attached to new episodes, an action–observation
temporal offset, a changed frame convention or unit in action extraction. This is my prior
favorite: it predicts exactly the observed signature, and it is the kind of thing that actually
happens when a collection setup runs for weeks — in the real DexYCB calibration corpus used for
Problem 1, the same physical cameras legitimately moved 14–25° between recalibration sessions, so
one episode pointing at the wrong session is silently catastrophic. *Confidence up:* the Problem 1
episode-QC tool flags calibration/sync failures concentrated in the new batch. *Confidence down:*
the new batch passes QC cleanly.

**H3 — Composition or recipe interaction.** The new batch shifts the training distribution (task
mix, operators, scenes, or sheer volume swamping the old data), or silently changes derived
statistics — the classic VLA gotcha being action-normalization stats recomputed over old+new,
which systematically re-scales every action the policy outputs at deployment. *Confidence up:*
batch-stats diff shows large distributional or normalization shift with clean QC. *Confidence
down:* distributions and stats match the old batch.

**H4 — Annotation/prelabel quality dropped in the new batch** (new annotators, a new prelabel
model version). Plausible, but broadly noisy labels should have moved offline metrics more than
"moderately," so it ranks below H2/H3. *Up:* human spot-check of a small new-batch sample shows
elevated label error. *Down:* spot-check error rate matches the old batch.

**H5 — Engineers A and B ("more data" / "bigger model").** Ranked last, deliberately. Success
dropped *because* data was added, so "more data" is a bet, not a diagnosis — more of a corrupted
batch makes things worse. A capacity limit does not switch on suddenly when data is added while
offline metrics hold. Neither claim is falsifiable within 48 hours, and both are the most expensive
options on the table. They return to consideration only if H1–H4 all come back clean.

## 2. Experiments (first 48 hours; ordered by information per hour)

**E0 (hours 0–2, no GPU): re-run the old 68% checkpoint on today's robot and protocol.**
*Change:* nothing — same weights, same task battery. *Measure:* task success over enough trials to
be conclusive: at these success rates the binomial 95% CI with n trials is roughly ±1.96·√(p(1−p)/n),
so ~30–40 trials per condition cleanly separates 68% from 41% (a 27-point gap). *Supports H1* if it
scores ~41%; *falsifies H1* (and validates the whole comparison) if it scores ~68%.

**E1 (hours 0–4, in parallel, no GPU): run the episode-QC tool (Problem 1) on the new batch vs. a
matched old-batch sample, plus a batch-stats diff** — task mix, episode lengths, which calibration
session each episode references, action-magnitude histograms, and the actual normalization
statistics used in training, old vs. new. *Supports H2* if failures (extrinsic mismatch, clock
offset, frame drops) concentrate in the new batch; *supports H3* if QC is clean but composition or
normalization stats shifted; *falsifies both* if the batches are statistically indistinguishable.

**E2 (the first of only two retrains): old data only, using the *current* training code and
config.** *Change:* remove the new batch, keep everything else as it is today. *Measure:* robot
success, same battery as E0. *Supports* "the new data is the problem" if it recovers to ~68%;
*falsifies it* — and implicates a training-code/config regression that shipped alongside the data —
if it stays low. This experiment is the single most decision-relevant retrain available.

**E3 (the second retrain, gated on E1/E2): old data + only the new episodes that PASS QC.**
*Measure:* robot success. *Supports H2* if success recovers most of the gap — and simultaneously
proves the QC gate is the remediation, not just the diagnosis. *Falsifies H2 in favor of H3/H4* if
success stays low even with only clean-looking new episodes.

**E4 (no training; robot time only): open-loop replay of a sample of new-batch demonstrations.**
*Change:* execute recorded action sequences verbatim on the robot. *Measure:* does the replayed
demonstration still accomplish its task? *Supports H2's action-side variant* (corrupted
retargeting/calibration in action extraction) if verbatim replays fail; *falsifies it* if replays
succeed — the recorded actions are fine and the problem is in what the model learned from
observations.

## 3. Data versus model (brief)

Isolate by layer, cheapest first: the QC tool separates sensor/calibration/sync problems from
everything else; a small human spot-check (the scarce review budget, spent only here) separates
annotation quality; E4's replay separates motion/trajectory quality from perception; E2/E3
ablations separate composition; temporal alignment shows up in the QC tool's cross-stream clock
checks. Only if every layer comes back clean does "the VLA itself" become the live hypothesis —
and at that point E2 has already told us whether the training recipe changed.

## 4. Multimodal models (brief)

A VLM/LLM accelerates triage: spot-checking at scale whether video content matches task labels,
flagging visually anomalous episodes for the QC queue, and summarizing config/log diffs between
collection weeks. I would explicitly *not* trust one, without a deterministic cross-check, for
anything geometric or temporal — calibration validity, synchronization offsets, 3D consistency, or
any numeric claim. A VLM asserting "this looks aligned" is exactly the silent false-accept that
Problem 2 warns costs more than a human review.

## 5. Decision

The rule is pre-committed before the evidence arrives, so the 48-hour deadline cannot pressure the
conclusion:

- **If E0 reproduces ~68%** (regression is real, eval is sound) **and E1 or E3 implicates the new
  batch** — the expected branch given the offline/online signature — then: **stop collection,
  repair the pipeline, and gate all re-ingestion through the episode-QC tool**, resuming collection
  only once newly collected episodes pass at the old batch's PASS rate.
- If **E0 also scores ~41%**, the data was never the problem: fix the robot/eval environment first
  and re-baseline before touching data or model.
- If **E2 stays low on old data alone**, revert to the last known-good training code/config and
  bisect — the batch is exonerated.
- Only if E1–E4 all come back clean do Engineer A/B's options (more data, bigger model) merit
  compute.

**Evidence threshold to commit:** the old-only retrain (E2) within the binomial CI of 68% over
≥30 trials, **and** the QC-gated retrain (E3) recovering at least half of the 27-point gap. Short
of that, the honest recommendation is the cheaper conditional one: keep collection paused (pausing
is nearly free; training on corrupted data is not) and spend the next 48 hours on the branch the
evidence points to.
