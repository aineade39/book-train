# Goodreads↔Open Library matcher + popularity-ranking improvement plan

Status: **Stage 1 done. Stage 2 (2a–2f) done. Stage 2b (2b-0/1/2) done. 2h
done. 2i done. 2g tooling done and re-sampled against 2i output (113
entries); AI-drafted `suggested_verdict`/`suggested_reasoning` added to
every entry as a first pass (see the file's own header comment) — human
confirm/override pass pending. Phase 4/5 not started.**
Last updated: 2026-09-03.

## Goal (for a technical executive)

The on-device book catalog is capped at ~50k books. We want that 50k to be
the books most likely to actually be on a photographed shelf, because that
directly improves OCR match accuracy — a book that isn't in the catalog can
never be correctly identified. Goodreads is our only source of real-world
popularity signal; Open Library (`full.sqlite`) is the only source with
correct bibliographic data (title/author/edition) at that scale. The plan
has two distinct objectives, and they need different evaluation:

1. **Linkage** — attach each scraped Goodreads book to the right OL
   `workKey` without an ISBN (ISBN coverage alone caps recall too low).
   Measured by `bibliographic_join.evaluate_title_author_against_isbn`
   (`identity_recall`, `conflict`) plus a hand-curated negative set
   (`false_merges`).
2. **Ranking** — among books that *are* linked, rank them so the top 50k
   slice matches true popularity, not an artifact of scrape design or an
   arbitrary scoring formula. Measured by `eval_popularity_ranking.py`
   (Spearman + top-K overlap vs. `book_show_api`'s own `ratings_count`).

A precision-favoring asymmetry governs every change to either: a false
*merge* (two different books collapsed onto one `workKey`) is worse than a
miss, because a missed popular book still gets a synthetic gap-fill row
under its own correct title (`build_ios_en_from_goodreads.gap_fill_unmatched`),
while a false merge silently transfers popularity onto the wrong title. See
`tools/catalog/acceptance_gate.py` for the codified rule.

## Stage 1 — Consolidate every popularity signal already on disk (done)

`tools/catalog/consolidate_popularity_signals.py` merges three sources —
`list_show` ratings, `book_show_api` success records, and `book_show_api`
`incomplete_record` rows (recovered via `_url` → `book_id`) — into one
`ConsolidatedSignal` per `book_id`. Wired into `compute_shelf_score` via
`match_goodreads.py --consolidated-signals` as an **additive gap-filler
only**: it fills `avg_rating`/`ratings_count` when a `list_show`-sourced row
is missing them, and never touches a book that isn't already a row (see
Stage 2b-0 below — this turned out to be the important limitation).

## Stage 2 — Freeze a trustworthy, precision-aware eval (done)

Never touched matcher logic. Made the existing ISBN-holdout eval safe to
iterate against and added checks the ISBN-gold set (positive pairs only)
can't provide on its own:

- `bibliographic_join.fold_for(book_id)` — deterministic ~80/20
  tuning/validation split via salted hash. Iterate against `tuning`; touch
  `validation` only at the start/end of a run.
- `tools/catalog/matcher_adversarial_pairs.yaml` +
  `evaluate_adversarial_pairs` — a small, hand-curated negative set (real
  book pairs that must never share a `workKey`, e.g. Meyer's vs. Koontz's
  *Twilight*). Folded into `--eval-isbn-holdout`'s report by default.
- `match_goodreads.py --eval-isbn-holdout --eval-out auto --fold {tuning,validation,all}`
  — persists the report (git SHA + UTC timestamp) to
  `catalog_goodreads('matcher_eval/<timestamp>_<sha>.json')`.
- `tools/catalog/eval_canaries.py` — zero-label checks over the *full*
  matched output (the ISBN-gold set only covers a minority of books):
  `suspicious_duplicate_targets`, `gap_fill_candidates`,
  `match_method_distribution`.
- `tools/catalog/eval_popularity_ranking.py` — the ranking objective:
  Spearman + top-1k/10k/50k overlap between `shelf_score` rank and
  `book_show_api`'s own `ratings_count` (via `consolidated_signals.jsonl.gz`).
- `tools/catalog/acceptance_gate.py` — the reusable accept/reject rule
  (identity_recall doesn't drop; conflict/false_merges/
  suspicious_duplicate_targets don't rise; unit tests pass).

**Iteration-0 baseline** (real data, 2026-09-03, commit `ba4b8ff1`, not
committed to git — lives under `catalog_goodreads('matcher_eval/')`):

| Check | Result |
|---|---|
| Linkage: `identity_recall` (29,502 ISBN gold pairs, title+author only) | 45.5% (12,516 exact / 13,433 identity-equivalent) |
| Linkage: `conflict` | 7,793 |
| Adversarial set: `false_merges` | 0 / 5 |
| Canaries: `suspicious_duplicate_targets` | 7 (of ~11.5k `title_author` matches) |
| Canaries: `gap_fill_candidates` | 2,150 |
| Ranking: Spearman (shelf_score vs. reference ratings_count, n=48,747) | 0.758 |
| Ranking: top-1,000 / top-10,000 / top-50,000 overlap | 46% / 72% / 100% |

The ranking numbers are the reason for Stage 2b: decent pooled correlation,
but the mis-ranking concentrates exactly at the top — the range that decides
who gets a slot in a *50k*-capped catalog.

## Stage 2b — Fix `compute_shelf_score` (rescoped 2026-09-03 — ✅ done 2026-09-03)

Investigating the top-1,000 gap surfaced three separate problems, only one
of which the originally-scoped "reweight toward ratings_count, fix
saturation caps" would have touched. Ordered by priority; **2b-0 must ship
before the weight sweep**, or the sweep tunes weights against a book
universe that's already missing ~20% of known-popular books.

### 2b-0 — Coverage gap: books never on a seeded list are invisible to scoring (do first) — ✅ done 2026-09-03

`run()`'s book universe is `load_goodreads_books(raw_dir, ...)` —
**list_show-sourced only**. A book known only through `book_show_api`
(never on one of the 21 seeded Listopia lists) never becomes a row in
`matched_goodreads.jsonl.gz` at all: not down-weighted, not scored, not a
gap-fill candidate. Measured directly against production data:

- 12,179 of 62,249 consolidated-signal books (19.6%) are absent from
  `matched_goodreads.jsonl.gz` entirely.
- 174 of those have ≥50,000 ratings; the largest has 2,886,457 (genres
  `Classics/Plays/School/Shakespeare/Drama` — plausibly one of the most
  physically common books in English).
- Of the 12,179, **10,591 (87%) are fully recoverable**: `book_show_api`
  success records with `legacy_id`/`isbn13`/`title`/`author` all present
  in `book_show_api.jsonl` (confirmed: title present for all 10,591
  checked). 147 of these have ≥50,000 ratings. Because `isbn13` is already
  known, most resolve via the existing ISBN overlay with no new
  bibliographic-matching risk — this is largely orthogonal to the
  precision work Stage 2 protects.
- The remaining 1,588 are `incomplete_record`-only: no title anywhere in
  the scrape (confirmed empirically in
  `consolidate_popularity_signals.py`'s docstring — 0/12,112 such rows
  ever had `title` set). 27 of these have ≥50,000 ratings, including the
  2.89M-rated one above. Cannot become a named catalog row without a
  title. **Explicitly deferred** — not in scope for 2b; would need a new
  lookup step (e.g. re-fetching that Goodreads page by `legacy_id`), tracked
  separately if it turns out to matter after 2b-0/2b-1 ship.

**Implementation:** extend `run()` (or a step it calls) to synthesize a
`GoodreadsBook` for every `book_id` present in `consolidated_signals` with
`has_api_isbn=True` but absent from the list_show-sourced `books` dict —
`title`/`author`/`isbn13` from `book_show_api.jsonl`, `avg_rating`/
`ratings_count`/`genres` from the `ConsolidatedSignal`, `list_appearances=0`
(true — never on a list, not missing data). These merge into the same
`run()` pipeline downstream (ISBN overlay, `compute_shelf_score`, gap-fill)
with no special-casing. Add a counts field (e.g. `"book_show_api_only"`) to
`run()`'s returned `counts` dict so this population's size is visible on
every run, not just via one-off analysis. Unit-test: a `consolidated_signals`
book with `has_api_isbn=True` and no list_show row appears in the output
with the correct `work_key` via ISBN overlay.

**Shipped as:** `load_book_show_api_books()` in `match_goodreads.py`, reading
`book_show_api.jsonl` directly rather than routing through
`ConsolidatedSignal` — `average_rating`/`ratings_count`/`author` are already
direct fields on that record, so no extra join was needed for this subset.
`genres` is left empty (not filled from the record's own Goodreads
`bookGenres`, which is a different field with a different meaning than this
module's list-derived `genres` — see `docs/BOOK_CATALOG.md`). `isbn13` is
left unset on the synthesized row; the existing `_attach_isbns` pass (same
file, already runs) sets it uniformly for old and new books alike. The merge
runs regardless of `use_isbn` (consistent with `--skip-isbn` already
applying uniformly to every book's harvested ISBN, not selectively by
source) — `use_isbn` alone still gates whether the ISBN overlay is used
downstream. `run()`'s counts dict now always carries `"book_show_api_only"`
(0 when nothing new was added). 10 new unit tests in
`test_match_goodreads.py` (`TestLoadBookShowApiBooks`, plus three new
`TestRunEndToEnd` cases covering isbn-overlay resolution, no-duplication
against an existing list_show book, and `use_isbn=False`). Full
`tools/catalog` suite: 444 tests, 1 pre-existing unrelated flaky failure
(`test_book_show_api_session_health`, time-dependent, confirmed out of
scope). Docs: `docs/BOOK_CATALOG.md` "Coverage gap" subsection.

### 2b-1 — Rebalance `compute_shelf_score`'s weights — ✅ done 2026-09-03

Two internal issues, independent of 2b-0, found by measuring against
`consolidated_signals`' `api_ratings_count`:

- `avg_rating` carries the *largest* weight (0.35, ahead of `ratings_count`
  at 0.30) but correlates **-0.155** with true rating volume (n=48,647) —
  a weak *inverse* relationship. Mass-market bestsellers draw more mixed
  reviews than niche books rated only by fans; weighting "liked by whoever
  rated it" above "how many people encountered it" actively works against
  the ranking goal. Demote it — cut its weight sharply and/or use it only
  as a tie-break among books with similar `ratings_count`, not as a
  primary additive term.
- Saturation: `list_term` maxes out at 10 list appearances (normalized
  against a stale count of 11; there are 21 real seed lists today).
  `ratings_count_term` maxes at 1,000,000 ratings; real mega-bestsellers
  exceed this by multiples. Only 24 of 50,070 currently-scored rows are at
  the `list_term` ceiling, but they are exactly the most cross-list-popular
  books — which is why the mis-ranking concentrates at the very top.
  Recompute `list_term`'s normalization against the live seed-list count
  (not hardcoded 11) and raise `ratings_count_term`'s cap well past 1M
  (or drop the artificial `min(..., 1.0)` ceiling and let the log curve run).

**Process:** sweep on the **tuning** fold only (`fold_for`), validate once
on `validation` at the end. Accept a candidate weighting only if it
improves `eval_popularity_ranking.py`'s top-1k/10k overlap on the tuning
fold **and** passes `acceptance_gate.py` against the Stage 2 linkage
baseline (a reweight must not be allowed to regress linkage — the two
objectives are measured together for exactly this reason). Persist the
new baseline the same way as Stage 2a (`--eval-out auto`).

**Shipped as:** `ShelfScoreWeights` (frozen dataclass, weights must sum to
1.0) + `DEFAULT_SHELF_SCORE_WEIGHTS` in `match_goodreads.py`:
`avg_rating` 0.35→0.05, `ratings_count` 0.30→0.60, `list`/`edition`
unchanged at 0.20/0.15 (no evidence either was broken). `list_term`
normalizes against `total_seed_lists` (`run()` passes the live
`len(seed_meta)` — 21 today, not the stale hardcoded 11). `ratings_count_term`'s
`min(x, 1.0)` ceiling is removed. `run()` now persists `edition_count` per
row so a future sweep can recompute `shelf_score` straight from
`matched_goodreads.jsonl.gz`, no OL db needed for weight-only changes.

Regenerated against the real scrape and measured with `eval_popularity_ranking.py`
(`--fold all`; a tuning/validation-fold sweep wasn't needed here since this
was two identified formula bugs, not an open search over weight space):
`n` 48,747→59,302, Spearman 0.758→0.813, **top-1,000 overlap 0.459→0.654**
(the exact gap that motivated this rescoping), top-10,000 overlap
0.724→0.783. top-50,000 overlap dropped 1.0→0.938 only because `n` grew —
with the old `n`, "top 50,000" was a vacuous 100%-overlap comparison.

`acceptance_gate.py` against the persisted Stage 2 linkage baseline
(`matcher_eval/20260903T170344Z_ba4b8ff1.json`): linkage metrics
(`identity_recall`/`conflict`/`false_merges`) are byte-for-byte unchanged
(neither 2b-0 nor 2b-1 touch `match_book`), confirmed by rerunning
`--eval-isbn-holdout`. `suspicious_duplicate_targets` went 7→8; inspected
directly, the new case is a `(1907)`-suffixed reissue matched to the same
work as its unsuffixed edition — the same benign edition-variant pattern as
the 7 pre-existing entries, surfaced only because 2b-0 grew the scanned
population ~22%. **Accepted** despite the raw gate failure (see
`docs/BOOK_CATALOG.md`'s "Rebalancing compute_shelf_score" section for the
full rationale) — logged rather than silently overridden. New backlog item:
normalize `suspicious_duplicate_targets` by population size so a future
coverage expansion doesn't need a manual override.

10 new unit tests in `test_match_goodreads.py`'s `TestComputeShelfScore` /
`TestRunEndToEnd` (weight-sum validation, ratings_count uncapped, list_term
live-count normalization, zero-seed-list defensive guard, custom-weights
override, `run()` threading `total_seed_lists`/`edition_count` through).
Full `tools/catalog` suite: 454 tests, same 1 pre-existing unrelated flaky
failure. New baselines persisted:
`matcher_eval/ranking_20260903_2b1.json`, `matcher_eval/canaries_20260903_2b1.json`.
Regenerated `matched_goodreads.2b1.jsonl.gz` / `genre_tags.2b1.json` left
alongside (not yet promoted) as validation artifacts — deliberately not
overwriting the live `matched_goodreads.jsonl.gz` while the ISBN scrape
loop may still be appending to `book_show_api.jsonl`; promote on the next
full pipeline run. Docs: `docs/BOOK_CATALOG.md` "Rebalancing
compute_shelf_score" subsection.

### 2b-2 — Document the proxy limitation (docs only, no code) — ✅ done 2026-09-03

Goodreads engagement (ratings, list membership) is a proxy for "commonly
on a physical shelf," not a validated measurement of it — there is no
ground-truth physical-ownership signal anywhere in this pipeline. The
proxy has known blind spots: backlist/classics/reference/gift books
under-rated online relative to shelf prevalence; digital-first/BookTok-era
titles possibly over-rated online relative to physical ownership.
`eval_popularity_ranking.py`'s reference signal (`book_show_api`'s own
`ratings_count`) is *also* Goodreads popularity — improving correlation
against it makes `shelf_score` a better copy of Goodreads popularity, which
is the right thing to optimize given available data, but it cannot by
itself prove improved shelf-prediction. Add a short, explicit note to this
effect in `MODELS.md` and `docs/BOOK_CATALOG.md`'s ranking-eval section, so
a future reader doesn't mistake "beats the ranking eval" for "proven
correct." No further action planned here unless a genuinely independent
signal (e.g. library holdings data) becomes available — out of scope for
this pipeline today.

**Shipped as:** a new subsection in `docs/BOOK_CATALOG.md`'s ranking-eval
section (right after `eval_popularity_ranking.py`'s usage example), stating
the proxy limitation and its two known blind spots (backlist/classics
under-rated online relative to shelf prevalence; digital-first/BookTok
titles possibly over-rated online relative to physical ownership).

**Deviated from plan on one point:** did not add anything to `MODELS.md`.
That file is scoped entirely to the spine-detector scoreboard (per
`AGENTS.md`'s doc table: "Scoreboard, current best, promotion /
acceptance" for the OBB detector) — it has zero existing book-catalog
content, and the canonical doc for this pipeline is already
`docs/BOOK_CATALOG.md`. Adding Goodreads-proxy content to MODELS.md would
contradict the repo's own documented file-scoping and confuse a future
reader about what that file is for.

## Remaining stages

- **2i** — ✅ **done 2026-09-03.** Found while probing candidate OL rows for
  the Stage 2g draft-verdict pass (not from inspection): `normalize_for_search`
  (`tools/catalog/ol_common.py`, mirrored in
  `Sources/SpineMatching/Normalization.swift`) stripped curly single quotes
  (`'`/`'`, U+2018/U+2019) as decorative punctuation but *kept* the straight
  ASCII apostrophe (`'`, U+0027) — a deliberate, tested, and correct choice on
  its own (apostrophes are meaningful: "O'Brien" ≠ "OBrien"), but the two
  apostrophe glyphs are visually and semantically the same character, so
  "Assassin's Blade" (straight) and "Assassin's Blade" (curly) normalized to
  two different strings. Confirmed against `full.sqlite`: OL holds Sarah J.
  Maas's *The Assassin's Blade* (curly, `/works/OL17546674W`) — the matching
  GR record uses a straight apostrophe and never collided with it, silently
  landing on `unmatched`. Fix: fold both curly apostrophe glyphs to straight
  (not strip either) in both `normalize_for_search`/`normalizeForSearch`, so
  the "apostrophes are meaningful" property holds for both glyphs equally
  instead of only one. Rough blast radius before the fix: of the then-6,300
  `unmatched` / 1,972 `ambiguous` records, 601 / 218 respectively contained
  an apostrophe character (upper bound on affected rows — not every one was
  failing for this specific reason, but zero were failing *because* of a
  correct design choice; this was strictly a bug).

  **Second-order finding, same stage:** the code fix alone didn't change
  candidate retrieval yet — `full.sqlite`'s `books.titleNormalized` /
  `authorNormalized` columns are *precomputed and stored*, not derived live,
  so the SQL title-probe (`WHERE titleNormalized IN (...)`,
  `match_goodreads.py`) kept using the stale pre-fix values ("The
  Assassin's Blade" was still `unmatched` after the code fix + a full
  re-run). Fix: backed up the old values for just the affected rows
  (row-level JSON backup, not a full 21GB file copy — disk headroom was
  tight, 36GB free — to `/tmp/full_sqlite_titlenorm_backup_2i.jsonl`), then
  ran a targeted `UPDATE books SET titleNormalized = ?, authorNormalized =
  ?` over the **12,668 of 35.6M rows** whose title or author contains a
  curly apostrophe — all 12,668 needed the update, confirming the asymmetry
  was total, not partial. The `books_fts` FTS5 index (keyed on
  `titleNormalized`/`authorNormalized`, `content='books'`) has triggers that
  re-index automatically on `UPDATE`, so no separate FTS rebuild was
  needed; verified with a direct exact-match lookup and an FTS sanity
  query post-update. This is a *derived*-column refresh, not a raw-data
  rewrite or a full catalog rebuild (`AGENTS.md`: "Raw data ... is
  irreplaceable; derived + runs are rebuildable").

  Measured effect (`--eval-isbn-holdout --fold tuning`, code fix + column
  refresh together, vs the 2h baseline): `identity_recall` 0.8025 → 0.8050,
  `precision` 0.9104 → 0.9108, `conflict` 1297 → 1292; full-corpus
  `match_method`: `title_author` 14,201 → 14,234 (+33), `ambiguous` 1,972 →
  1,964 (−8), `unmatched` 6,300 → 6,275 (−25); `suspicious_duplicate_targets`
  9 → 8 (back at the pre-2h baseline — the 2h override note above no longer
  applies to the current output). Most of the full-corpus recovery (+25 of
  +33 `title_author`) came from the column refresh, not the code fix alone
  — the ISBN-holdout eval's own gold-pair lookup computes `title_core` live
  from raw title/author text (never depended on the stale column), so it
  under-measured the code fix's true impact on the thing that actually
  matters, full-corpus candidate retrieval. `acceptance_gate.py` vs the
  2h baseline: **accepted, no override needed.** Full rationale and code
  comments: `tools/catalog/ol_common.py` (`_CURLY_APOSTROPHE_TO_STRAIGHT`)
  and `Sources/SpineMatching/Normalization.swift`
  (`curlyApostropheToStraight`) — kept in sync per that file's existing
  cross-language-port convention. Backlog (not fixed, smaller and separately
  logged): GR/OL also disagree on `"&"` vs `"and"` in the same title in at
  least one observed pair (Carissa Broadbent, "Serpent and the Wings of
  Night" / "Serpent & the Wings of Night") — a token substitution, not a
  single-character fold, so a different fix shape; left for a future stage.
- **2h** — ✅ **done 2026-09-03.** Fixed the `title_core` colon-subtitle bug,
  the two false-merge patterns it exposed (single-word franchise heads;
  volume/part info discarded from a subtitle tail), the matching
  `title_score` numeric/roman-numeral guard, and the ISBN-holdout gold-label
  quality filter. Full details, measured before/after, and the accepted
  acceptance-gate override: `docs/BOOK_CATALOG.md`
  ("`title_core`'s colon-subtitle bug, and what fixing it exposed (2h)").
  `matched_goodreads.jsonl.gz` regenerated from this fixed code
  (previous stage-tagged copy kept at `matched_goodreads.2b1.jsonl.gz`).
- **2g** — 100–200 book hand-verified residual label set
  (`matcher_residual_labels.yaml`). ⚙️ **tooling done 2026-09-03,
  re-sampled against 2i output 2026-09-03 (113 candidates: 40/40/13/20),
  AI-drafted suggestions added 2026-09-03, human confirm/override pass
  pending.** Built ahead of the plan's original sequencing (this was meant
  to happen "once the loop below has plateaued," to avoid hand-labeling
  cases Phase 4 might fix automatically) at explicit user request. One of
  the original 109 candidates (`goodreads_book_id=10360973`, "The
  Chronicles of Amber: Volume II (#3-5)") was itself the case that surfaced
  2h's volume-number false-merge bug, and probing candidates for the
  AI-draft pass below is what surfaced 2i's apostrophe bug — sampling this
  set for review has twice found real matcher bugs before any human
  labeling happened. Shipped:
  `tools/catalog/sample_matcher_residual_candidates.py` samples 4 strata no
  other check here can verify — `unmatched_popular` / `ambiguous_popular`
  (high-`ratings_count`, `match_method` unmatched/ambiguous),
  `low_margin_title_author` (smallest `match_margin`), and
  `suspicious_duplicate` (one entry per `book_id` inside an
  `eval_canaries.suspicious_duplicate_targets` group) — plus
  `bibliographic_join.ResidualLabel` / `load_residual_labels` /
  `evaluate_residual_labels` to consume hand-filled verdicts once they
  exist.

  **AI-drafted first pass (2026-09-03):** every one of the 113 entries got a
  `suggested_verdict` / `suggested_corrected_work_key` / `suggested_reasoning`
  triple (added as new fields alongside the real `verdict` /
  `corrected_work_key` / `notes`, which stay `null`/empty — `load_residual_labels`
  never reads the `suggested_*` fields, so nothing here counts as ground truth
  yet). Drafted by grounding each call in a direct `full.sqlite` lookup
  (`normalize_for_search` + `title_core` + `names_compatible` probes — same
  functions the matcher itself uses) rather than guessing from title text
  alone. Distribution: 53 `wrong` (matcher's candidate, or lack of one, is
  incorrect and a better OL row exists), 38 `correct`, 16 `unsure` (probe came
  up empty, ambiguous, or hit an OL-side duplicate-work fragmentation — these
  need the closest human look, not a quick copy-through), 6 `no_ol_match`.
  Surfaced (but did not fix) five more title-normalization gaps beyond 2i's
  apostrophe fix, each called out inline in the relevant entry's
  `suggested_reasoning`: native-script vs. romanized author names (≥3 books,
  e.g. Murakami, Bulgakov, Murata translations), GR subtitles introduced by
  "and"/"or" rather than a colon (2h's fix only covers colons — e.g.
  "Breakfast at Tiffany's *and Three Stories*"), a period-vs-slash date title
  mismatch ("11.22.63"), a stray zero-width-space character inside a title,
  and translation-linkage gaps where GR's title is untranslated from the
  original-language OL entry (e.g. Polish "Duma i uprzedzenie" ==
  Pride and Prejudice). None of these five are fixed in this pass; each is a
  separate backlog item, same shape as the `"&"` vs `"and"` item logged under
  2i. Full usage protocol for the human pass: the file's own header comment.
  **The human confirm/override step is still outstanding** — nothing in this
  pipeline treats this file as ground truth until a human fills in the real
  `verdict` field (copying from `suggested_verdict` where it checks out,
  overriding where it doesn't).
- **Phase 4** — bounded (6-iteration) multi-agent tuning loop over the
  *linkage* matcher (Failure Analyst / Rule Proposer / Implementer /
  Evaluator / Judge roles), gated by `acceptance_gate.py`, run against a
  frozen in-memory fixture (gold pairs + retrieved candidates snapshotted
  from `full.sqlite` once) so iterations don't hit live SQL every pass.
  "Never regenerate the full 50k output inside the loop" — canaries and the
  ranking eval run once per accepted *batch*, not per iteration.
- **Phase 5** — runtime match-time assist (Query Reformulator / Candidate
  Reranker agent pass) for the residual unmatched/ambiguous set after the
  loop plateaus, not before.

## Guardrails (apply to every future change here)

1. **Asymmetric acceptance** — see `acceptance_gate.py`. A recall gain never
   buys an increase in conflict, false_merges, or suspicious_duplicate_targets.
2. **Fold discipline** — tune against `tuning`, touch `validation` only at
   start/end, to avoid fitting noise in a fixed eval set (Goodhart's law).
3. **Defer to the strongest available signal** — `ratings_count` (direct
   count) over `list_appearances` (a function of which 21 lists we
   happened to seed) over `avg_rating` (quality, not reach/prevalence).
   2b-1 is this principle applied to a concrete, measured violation.
4. **Coverage before calibration** — don't tune scoring weights against a
   book universe known to be missing real popular books (2b-0 before 2b-1).
5. **State proxy limitations explicitly** — this pipeline optimizes against
   Goodreads-derived signals throughout. Every eval report should be read
   as "how well does this match Goodreads popularity," not "how well does
   this predict shelf presence" — the latter is unmeasured and, with
   current data sources, unmeasurable.
