# PersonaMem-v2 32k funnel analysis — 2026-10-06

Scope: `scripts/run_experiment.sh` on `data/prepared/personamem-v2-32k.jsonl`
(conversation 1, 26 QAs, 14 add chunks), followed by a gold-metadata funnel
audit and architecture changes grounded in `llm-memory-survey`.

## 1. Runs

| Run | Dir | Score |
|---|---|---|
| Baseline (HEAD) | `runs/experiment-20261006-baseline` | 14/26 = 53.85% |
| Round 1 (S1 + S5 fixes) | `runs/experiment-20261006-improved` | 14/26 = 53.85% |
| Round 2 (+ coverage witnesses) | `runs/experiment-20261006-improved2` | **15/26 = 57.69%** |
| Held-out conv 2, baseline code | `runs/heldout-baseline` | 17/30 = 56.67% |
| Held-out conv 2, improved code | `runs/heldout-improved` | **18/30 = 60.00%** |

Held-out = `data/prepared/personamem-v2-32k-heldout-conv2.jsonl` (prepared
line 2, persona never looked at during analysis, 30 QAs). It is the
generalisation check: no change below was tuned on it.

## 2. Funnel method

`scripts/funnel_gold_snippet.py --run RUN_DIR [--all] [--compare RUN_B]`.
For every QA the CSV gold metadata (`related_conversation_snippet`,
`preference`) identifies the exact user turn that carries the answer. The
tool then follows that turn through the pipeline:

| Stage | Question asked of the logs |
|---|---|
| S0 | was the snippet turn ingested (`source_messages`)? |
| S1 | did any stored memory cite the snippet turn (`extraction_trace`)? |
| S2 | did that memory enter the fused/ranked pool (`search-debug`)? |
| S3 | did it survive rerank into the packet / source catalog? |
| S4 | does the gold option have a claim with a valid citation (`choice_alignment`)? |
| S5 | is the gold option in `answer_eligible_options`? |
| S6 | did the selector pick it? |

Losses are attributed to the first failing stage.

### Baseline funnel (12 wrong)

| Stage | QAs | Dominant cause |
|---|---|---|
| S0 | 5 | snippet turn in a chunk the plan never ingested |
| S1 | 1, 4, 10, 17, 18, 20, 23, 24 (+ 4 correct-by-luck) | no memory cites the gold turn |
| S3 | 8 | rerank dropped |
| S4 | 2 | gold option judged unsupported |
| S6 | 13 | wrong pick among eligible |

Root causes behind S1 (12 QAs incl. correct-by-luck):

1. `_verify_semantics` (LLM triple verifier) rejected 102 facts, mostly
   *interest* facts inferred from the user's own question ("If I had a tiny
   garden…", "why do lakes feel warmer in July"). The extraction prompt rule 8
   explicitly asks for those facts, so the pipeline contradicted itself.
2. 34 `Evidence quote does not match` rejections: 6 were verbatim quotes filed
   under the wrong `message_index`, the rest paraphrase/normalisation.
3. Answer side: `CHOICE_ALLOW_INFERRED=False` dropped the `inferred` tier
   whenever any `supported` option existed, so a distractor citing a true but
   unrelated trait (cycling, Jayhawks) always beat a topically-relevant gold
   option that was only *inferred* from the user's questions.

### Round-2 funnel (`--compare baseline improved2`) — 15/26

Gained QA2, QA8 (coverage witness surfaced the Facebook/professional-email
source), QA13 (selector now prefers the topical option). Lost QA14, QA15: the
support stage still cites the right user turn ("I used to ride there a lot
… river trail"), but the entailment judge (prompt 15) flipped from
`personal_supported` to `unsupported_personal` on identical evidence across
all three runs after round 1 — the claim text adds "connected with freedom",
which a strict reading does not find in the quote. This is judge strictness
variance, not a regression introduced by the new code paths (the claim,
citation and catalog entry are byte-identical between runs).

### Round-1 funnel (`--compare baseline improved`)

S1 losses fell from 12 to 6 (interest facts are now stored, e.g. "The user is
interested in small-space gardening"), but the losses moved downstream to
S4/S5: the gold options were in the catalog yet the support judge cited the
wrong user source (QA23: a surfing story instead of the lake-temperature
question that retrieval had already ranked #1 for option A) or the entailment
stage returned `not_entailed`. Net score unchanged (gained 2, lost 10, 14, 15
to LLM variance / changed catalogs).

## 3. Changes

### S1 — extraction (`app/add_pipeline.py`, `prompts/01_extract_amu.txt`)

* `_verify_semantics` prompt gained an **INTEREST EXCEPTION**: a user's own
  question, remark or first-person hypothetical establishes interest-level
  facts; triples are projections, so reject only when they add entities or
  reverse meaning.
* New `_repair_evidence_indices`: when a quote is verbatim but filed under
  the wrong index and appears in exactly one other message of the batch, the
  index is re-pointed (trace event `evidence_index_repaired`). Attribution
  (`epistemic_status`) follows the repaired source.
* Prompt rule 8: first-person hypotheticals → `preference`/interest fact,
  never ownership; the quote must be the user's own turn.

### S5 — eligibility (`app/answer_choice.py`, `app/config.py`, `prompts/14_choice_select.txt`)

* `CHOICE_ALLOW_INFERRED` default on; `inferred` options compete with
  `supported`/`partial` ones instead of being discarded.
* Selector prompt: *RELEVANCE OVER LABEL* — prefer the option anchored in the
  user's own turns on the question's topic over a supported-but-unrelated
  profile trait.

### S4 — witness selection (`app/choice_witness.py`, flag `CHOICE_COVERAGE_WITNESSES`)

* `supplement_from_coverage`: retrieval already tags returned memories with
  `coverage_ids` (`option:i`) from the per-option queries. For each option
  whose semantic witness missed, the best sentence of the top user-role card
  tagged for that option is prepended as a witness hint
  (`coverage_rescue=True`, capped at `MAX_PER_OPTION`). Offline replay on the
  round-1 logs reached the gold option in ~10 QAs (QA2, 4, 7, 8, 11, 12, 15,
  22, 23).

### Deliberately **not** changed

The evidence ladder still rejects gap-2 inferences (interest → ownership /
habit, `interest_does_not_prove_trait`). Several dataset gold options (QA18
"spice rack", QA24 "garden") require exactly that jump. Allowing it would
raise this persona's score but reintroduce the hallucination class the ladder
exists to block, so it is left strict; these QAs remain S4/S5 losses by design.

## 4. Survey grounding

* `notes/05-evermemos.md`: profile consolidation of *stable traits vs
  temporary states* lifted PersonaMem-v2 43.93 → 53.25; interest statements
  from the user's questions are exactly the stable-trait signal our verifier
  was throwing away.
* `notes/20-memoryos.md`, `notes/18-memorybank.md`: user traits (evolving
  interests, habits) are a first-class memory tier derived from dialogue, not
  only from explicit self-reports.
* `notes/22-memos.md`: retrieval scheduling should feed the reasoning stage
  the evidence it already found — the motivation for reusing `coverage_ids`
  as witness hints rather than a second retrieval.

## 5. Generalisation

All changes are rule-level (verifier contract, index repair, tier policy,
reuse of retrieval coverage) and contain no persona, topic or option text.
The held-out conversation above was run with both code bases to measure the
effect outside the analysed persona: **17/30 → 18/30** (+1, no category
regressed by more than one QA; `funnel_gold_snippet.py --compare
runs/heldout-baseline runs/heldout-improved`). Gains: QA51, QA53 (gold now
supported via coverage witnesses); loss: QA39 (entailment judge variance).
The direction matches the analysed persona (+1), so the changes are not
fitted to one conversation, but the effect size is small relative to
single-QA LLM variance; a multi-conversation run is needed for a tight
estimate.

Held-out funnel also shows 7 QAs lost at **S0** (gold turn never ingested
because the eval plan caps add chunks) and ~12 at **S4**. S0 is an eval-plan
limit, not a memory-system defect; S4 is the next target.

## 6. Tests

`scripts/run_selftests.py`: the suites touched here pass
(`selftest_evidence`, `selftest_answer_choice`, `selftest_choice_witness`).
New cases: wrong-index repair + unlocatable-quote fallback, inferred tier
competing with an unrelated verified trait under both flag values, and four
coverage-witness cases (rescue, dedupe, prepend/cap, assistant-only → none).
`test_month_range_recall_and_answer_metadata` and nine other suites fail
identically at HEAD and are unrelated.

## 7. Reproduce

```bash
cd memory_system
AML_EXPERIMENT_DIR=runs/<name> bash scripts/run_experiment.sh
.venv/bin/python scripts/funnel_gold_snippet.py --run runs/<name> --all
.venv/bin/python scripts/funnel_gold_snippet.py --compare runs/<a> runs/<b>
# held-out
AML_EVAL_DATA=./data/prepared/personamem-v2-32k-heldout-conv2.jsonl \
AML_EXPERIMENT_DIR=runs/heldout-<name> bash scripts/run_experiment.sh
```
