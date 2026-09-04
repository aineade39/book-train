# Book catalog (Open Library)

Build on-device SQLite catalogs for spine OCR matching. Matching logic lives in
`SpineCatalog` + `SpineMatching` per [`BOOK_ID_IOS_PIPELINE.md`](BOOK_ID_IOS_PIPELINE.md)
— this doc covers **catalog content only**.

| Doc | Role |
|---|---|
| [`DATA.md`](../DATA.md) | Data root paths |
| [`AGENTS.md`](../AGENTS.md) | Command cheat sheet |
| [`BOOK_ID_IOS_PIPELINE.md`](BOOK_ID_IOS_PIPELINE.md) | Runtime matching architecture |

## Quick start (fixture smoke — no OL download)

```bash
# Build dev catalog from committed fixture + install into book-id-ios
python tools/build_book_catalog.py --profile dev_smoke --fixture Tests/fixtures/ol-mini --install-ios

# Match test (no photo/model)
swift run -c release book-match "dune frank herbert" \
  --db $BOOK_SPINES_DATA/derived/book-catalog/dev_smoke.sqlite --json
```

## Production build (full pipeline)

```bash
# Download OL dumps → ETL → all profiles (full, ios_en, dev_smoke)
python tools/build_book_catalog.py --all --install-ios

# One profile, reuse cached intermediate
python tools/build_book_catalog.py --profile ios_en --reuse-intermediate --install-ios

# Fast subset from existing full DB (no OL re-parse)
python tools/build_book_catalog.py --profile ios_en \
  --subset-from $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite
```

## Phase 1 — CSV path (manual / small lists)

```bash
python tools/catalog/ol_to_csv.py --languages eng --max-works 250000 -o /tmp/works.csv
swift run -c release catalog-build /tmp/works.csv --db ios_en.sqlite
```

## Data layout

```text
$BOOK_SPINES_DATA/
  raw/open-library/              # OL dumps + SOURCE.md
  derived/book-catalog/
    intermediate/
      manifest.json
      works.jsonl.gz
      isbns.jsonl.gz             # reserved for future ISBN index
    full.sqlite + full.json
    ios_en.sqlite + ios_en.json
    dev_smoke.sqlite
```

Profiles: [`tools/catalog/profiles.yaml`](../tools/catalog/profiles.yaml) (JSON mirror for stdlib-only).

### `ios_en_shelf` (Goodreads-informed re-rank + gap-fill)

Same shape as `ios_en`, but `popularityRank` is re-ranked (and gap-filled with a
handful of popular-but-OL-absent titles) using Goodreads Listopia shelf signals
instead of raw OL edition count alone — see `tools/scrape_goodreads_lists.py` /
`tools/catalog/match_goodreads.py` / `tools/catalog/build_ios_en_from_goodreads.py`.

Has a `custom_build_script` in profiles.yaml/json and is **not** built by
`build_book_catalog.py` (that generic `--subset-from` path can't apply a language
filter to a subset build at all, and knows nothing about Goodreads):

```bash
# One-time: build full.sqlite if it doesn't exist yet
python tools/build_book_catalog.py --profile full --install-ios=false

# Scrape (infrequent, checkpointed/restartable — see the script's docstring)
python tools/scrape_goodreads_lists.py

# Match scraped books against full.sqlite (title+author, then harvested ISBN overlay)
python tools/catalog/match_goodreads.py --ol-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --book-show-api $BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl \
  --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz

# Title+author only (no ISBN). ISBN-holdout eval of that matcher:
python tools/catalog/match_goodreads.py --ol-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite --skip-isbn
python tools/catalog/match_goodreads.py --ol-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --book-show-api $BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl \
  --eval-isbn-holdout

# Rebuild ios_en_shelf.sqlite from a scratch copy of full.sqlite (never mutates full.sqlite itself)
python tools/catalog/build_ios_en_from_goodreads.py \
  --full-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --intermediate-dir $BOOK_SPINES_DATA/derived/book-catalog/intermediate \
  --matched-goodreads $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz \
  --output $BOOK_SPINES_DATA/derived/book-catalog/ios_en_shelf.sqlite
```

#### GR↔OL join (title+author, then ISBN overlay)

The on-device catalog is capped (~250k). The 50k most popular Goodreads books
are the ones we most need inside that cap, because they are the books most
likely to appear on a photographed shelf. ISBN is the clean OL join key, but
`list_show` does not carry ISBNs and the `book_show_api` harvest will never
cover every popular title (no ISBN in GR, incomplete scrape, OL missing that
ISBN). Gap-fill can insert an unmatched GR title as a synthetic row, but that
does not attach GR popularity to the canonical OL work — so the real OL row
can still lose a subset slot.

`tools/catalog/bibliographic_join.py` is the non-ISBN join. Inputs are only
`list_show` title/author and `full.sqlite` `books` (title, author,
editionCount). Publisher is the standard third bibliographic key; it is not
on either side of this join (`list_show` lacks it, `books` does not store it),
so it is not used. Author last-name compatibility is the same-title
disambiguator (Meyer vs Koontz *Twilight*). Edition count is the tie-break
among OL duplicate works of the same identity.

Do **not** reuse the on-device OCR accept policy (`token_set_ratio`, 90/8)
for this join. That scorer saturates at 100 when one clean title is a token
subset of another, so "The Hunger Games" ties its own companion/guide and
the margin test marks the most popular GR books `ambiguous`. Measured on
the previous matcher: 32,077 GR books → 14,269 fuzzy / 15,866 ambiguous /
1,942 unmatched.

Bibliographic rules:

| Step | Rule |
|---|---|
| Title core | Strip series `(… #n)`, leading article, subtitle after `:`, treat `-` as space |
| Year titles | `1984` ≡ `nineteen eighty-four` |
| Author parse | `and`/`&` → multi-author; one comma → catalog-inverted `Last, First`; 2-token names also try order-flip (Liu Cixin / Cixin Liu) |
| Blocking | Exact ``titleNormalized`` probes (indexed) plus a long title-prefix LIKE fallback; author gate in Python. Does not load ``full.sqlite`` into RAM. |
| Author gate | Last-name overlap, 2-token order-flip, or maiden-token containment |
| Identity accept | Same title core (or year alias) + author gate → keep highest `editionCount` |
| Strong accept | `token_sort_ratio` ≥ 90 and margin ≥ 5 against the next *distinct* title core |
| ISBN overlay | If a harvested ISBN hits `book_isbns`, that `workKey` wins |
| Coverage | `--book-show-api` records also add wholly new books, not just an ISBN overlay for existing ones — see below |

Evaluate the bibliographic matcher with `--eval-isbn-holdout`: hide ISBN from
the matcher, use harvested ISBNs that hit `book_isbns` as gold, report exact
`workKey` recall plus identity-equivalent recall (same title core + compatible
author — the right answer when OL split one work across several keys).

`match_method` values: `title_author`, `isbn`, `ambiguous`, `unmatched`.
`build_ios_en_from_goodreads.py` still treats only `unmatched`/`ambiguous`
as gap-fill candidates.

#### Coverage gap: books never on a seeded list (fixed)

`run()`'s book universe used to be `load_goodreads_books(raw_dir, ...)` —
**list_show-sourced only**. A book known only through `book_show_api`
(never on one of the 21 seeded Listopia lists, which were picked to get
broad coverage, not because absence from them means anything about
popularity) never became a row in `matched_goodreads.jsonl.gz` at all: not
down-weighted, not scored, not a gap-fill candidate — silently invisible to
the whole pipeline. Measured against a real scrape before the fix: 12,179
of 62,249 known `book_id`s (19.6%) were missing this way, 174 of them with
≥50,000 ratings, the largest with 2,886,457 (genres
`Classics/Plays/School/Shakespeare/Drama`).

`load_book_show_api_books` (`match_goodreads.py`) now reads `--book-show-api`
directly and returns a full `GoodreadsBook` for every record with
`legacy_id`+`isbn13`+`title`+`author` all present — the recoverable 87% of
that gap (10,591 of the 12,179; confirmed title present for every one
checked). `run()` adds these for any `book_id` not already present via
`list_show`, with `list_appearances=0` (true — never on a list, not missing
data) and `genres=set()` (this field means list-derived tags, not
Goodreads' own per-book `bookGenres` — a book with zero list appearances
legitimately has zero list-derived tags). Because `isbn13` is already
known, these resolve via the existing ISBN overlay with no new
bibliographic-matching risk. `run()`'s returned counts dict always includes
a `book_show_api_only` key so this population's size is visible on every
run.

The remaining 1,588 `book_id`s (13% of the gap, 27 with ≥50,000 ratings,
including the 2.89M-rated one above) are `_scrape_warning: incomplete_record`
rows with no title anywhere in the scrape (confirmed empirically — 0/12,112
such rows ever had `title` set; see `consolidate_popularity_signals.py`).
They cannot become a named catalog row without a title and are not handled
here — a separate, harder problem if it turns out to matter later.

#### Rebalancing `compute_shelf_score` (2b-1)

Two formula bugs, found the same way as the coverage gap — by measuring
against `consolidated_signals`' `api_ratings_count` rather than trusting the
formula's shape:

- `avg_rating` carried the *largest* weight (0.35, ahead of `ratings_count`
  at 0.30) but correlated **-0.155** with true rating volume (n=48,647) — a
  weak *inverse* relationship. Mass-market bestsellers draw more mixed
  reviews than niche books rated only by fans, so weighting "liked by
  whoever rated it" above "how many people encountered it" actively worked
  against the ranking goal.
- `list_term` normalized against a hardcoded `11`, stale since the seed
  list count grew to 21; `ratings_count_term` hard-capped at `min(x, 1.0)`,
  tying every book past ~1M ratings on that term — exactly the range a
  50k-capped catalog cares most about getting right (only 24 of 50,070
  scored rows hit that ceiling, but they were the most cross-list-popular
  books, so the mis-ranking concentrated at the very top).

Fix, in `ShelfScoreWeights` / `compute_shelf_score` (`match_goodreads.py`):
`avg_rating` demoted `0.35 -> 0.05`, `ratings_count` promoted `0.30 -> 0.60`
(the strongest true signal absorbs the weight `avg_rating` gave up);
`list_term` normalizes against `total_seed_lists` (`run()` passes the live
`len(seed_meta)`, not a constant); `ratings_count_term`'s ceiling is
removed, so a mega-bestseller can score above 1.0 rather than tying with
every other book past 1M ratings. `list`/`edition` weights (0.20/0.15) are
unchanged — no evidence either was broken, and reweighting them without
measurement is exactly the kind of untested change this stage is trying to
avoid. `run()` now also persists `edition_count` on every output row, so a
future reweight sweep can recompute `shelf_score` straight from
`matched_goodreads.jsonl.gz` without a live OL db connection.

Measured effect, regenerating against the real scrape (`eval_popularity_ranking.py`, `--fold all`):

| | before (pre-2b-0/2b-1) | after (2b-0 + 2b-1) |
|---|---|---|
| scored books (`n`) | 48,747 | 59,302 |
| Spearman | 0.758 | 0.813 |
| top-1,000 overlap | 0.459 | **0.654** |
| top-10,000 overlap | 0.724 | 0.783 |

The top-1,000 overlap gain (the exact gap flagged when this rescoping
started) is the headline number: it's the range that decides who gets a
slot in a 50k-capped catalog. top-50,000 overlap *dropped* (1.0 -> 0.938)
only because `n` grew — with 48,747 total books, "top 50,000" was a vacuous
100%-overlap comparison; with 59,302 it's a real one. Not a regression.

**Acceptance-gate note:** `acceptance_gate.py` flagged one new
`suspicious_duplicate_targets` case (7 -> 8) versus the persisted Stage 2
baseline. Inspected directly: it's a `(1907)`-suffixed reissue of "Forty
Singing Seamen And Other Poems" by Alfred Noyes matched to the same work as
the unsuffixed edition — the same benign edition-variant pattern as the
7 pre-existing entries (e.g. "Fahrenheit 451" / "Farenheit 451"), not a
matching-quality regression. It appeared *only* because 2b-0 added ~10.6k
new books to the population the canary scans — a ~22% population increase
producing one incidental collision is proportionally *better* than the
baseline rate, not worse. The gate's raw-count check has no way to
distinguish "population grew" from "matcher got worse," which is exactly
the kind of blind spot this pipeline's guardrails exist to catch — flagging
it here rather than silently overriding it. Judgment call: **accepted**
despite the raw gate failure, given (a) linkage metrics (`identity_recall`,
`conflict`, `false_merges`) are byte-for-byte unchanged — neither 2b-0 nor
2b-1 touch `match_book` — and (b) the flagged case is a confirmed benign
edition variant, not a new false-merge pattern. Backlog: normalize
`suspicious_duplicate_targets` by population size (a rate, not a raw count)
so a future coverage expansion doesn't require a manual override.

#### `title_core`'s colon-subtitle bug, and what fixing it exposed (2h)

Found while sampling `unmatched_popular` candidates for Stage 2g: dozens of
famous, high-`ratings_count` English-language books ("Atomic Habits",
"Sapiens", "Norwegian Wood") had **zero** OL candidates, even though OL
holds the exact title/author under a plain, short title. Root cause in
`bibliographic_join.title_core`: the colon-subtitle split
(`"Atomic Habits: An Easy & Proven Way..." -> "atomic habits"`) checked for
`":"` on the string returned by `normalize_for_search`, which had already
stripped `":"` as decorative punctuation — so the split's `if ":" in
normalized` never fired for *any* colon-subtitle title (the single most
common nonfiction-subtitle convention). The GR title kept its full
subtitle, OL's own catalog title usually didn't, so the two never became
`titles_identity`-equal and the SQL title-probe (whose keys come from this
function) never even pulled OL's short-titled record into the candidate
pool — silently `unmatched`, not `ambiguous`. Affected **~4,400 of 60,652
GR books (7%)** — every colon-subtitled title landing on `unmatched` or
`ambiguous`.

Fixing the check alone regressed two *different* cases, both found by
sampling real output after the fix, not by inspection alone:

1. `title_core` now also correctly recognizes "Batman: Knightfall, Part
   Three: Knightsend" as colon-subtitled and stripped it to bare
   `"batman"` — but "Batman:" here is a franchise-prefix, not a
   title+subtitle, and "batman" alone `titles_identity`-matches whatever
   OL work happens to be titled just "Batman", conflating unrelated
   stories. Fix: only strip when the head has `>= 2` words — filters out
   single-word franchise prefixes ("Batman", "Ghostbusters") while
   keeping every multi-word case ("Atomic Habits", "Guns, Germs, and
   Steel"). Cost: genuinely single-word main titles with a subtitle
   ("Sapiens: A Brief History of Humankind") also don't get stripped — a
   narrower, accepted gap versus the false-merge risk.
2. A substantial multi-word head can still discard essential info if it's
   the *tail* that carries it: "The Chronicles of Amber: Volume II
   (...#3-5)" has a solid 4-word head ("The Chronicles of Amber", passes
   the check above) but "Volume II" is the only thing distinguishing it
   from "Volume I" — this exact pair was already flagged as a suspected
   matcher bug in this plan before this investigation found the root
   cause. Fix: also don't strip when the discarded tail contains a
   volume/part keyword or a small standalone number (digit, spelled-out,
   or roman numeral — `Volume II` / `Part Three` / `Year One` all count).

Recovering that same long tail also surfaced a third, unrelated
pre-existing weakness: `title_score`'s `token_sort_ratio` treats one
differing digit (or roman numeral) among many shared tokens as a trivial
edit, so "Volume 2" vs "Volume 3" (or "Volume I" vs "Volume II", "Part Two"
vs "Part Three") scored above `TITLE_ACCEPT` despite being different
specific installments (Swamp Thing, Ghostbusters, Batman: Knightfall, and
Chronicles of Amber volumes all collapsed onto one candidate this way).
Fix: `bibliographic_join._standalone_numbers` extracts small cardinal
numbers (digit, spelled-out, or roman numeral) that appear as their own
token; `title_score` now forces `0.0` when both cores have a number and
the sets are disjoint. (Bare "i" is deliberately included despite
colliding with the pronoun "I" once lowercased — a title core is a noun
phrase, not a sentence, so the collision is rare, and the failure mode
this guards against is worse than the occasional over-cautious rejection.)

Measured effect (`--eval-isbn-holdout --fold tuning`, `full.sqlite`):

| | before | after (all four fixes) |
|---|---|---|
| `identity_recall` | 0.457 | **0.803** |
| `precision` (1 - conflict rate) | 0.736 | **0.910** |
| `conflict` | 6,252 | 1,297 |
| `false_merges` (adversarial) | 0 | 0 |
| full-corpus `match_method` | `title_author` 12,518 / `ambiguous` 2,273 / `unmatched` 7,682 | `title_author` 14,201 (**+1,683**) / `ambiguous` 1,972 (**-301**) / `unmatched` 6,300 (**-1,382**) |

**A fifth, retroactive finding while confirming the above:** the raw
`conflict` count *increased* (6,252 -> 8,304) on the very first re-run,
before the `title_score`/`title_core` guards above existed — recovering
previously-`unmatched` books looked like a precision *regression*.
Sampling the new conflicts showed the opposite: **93% had a gold-work
author that isn't even `names_compatible` with the querying book's own
author** — the "gold" ISBN, per OL's own `book_isbns` table, points to an
unrelated or placeholder work (`"Untitled"`, author `"To Be Announced"` —
publishing pre-registers ISBNs before a book's final title/author is set,
and some of those OL stub records never get updated). This is a property
of the long tail generally, not of this one change: OL's ISBN->work
linkage is noisiest exactly where matching is hardest, so *any* future
recall gain into previously-`unmatched` territory will trip this same
false alarm. Fixed at the eval level, not by manual override each time:
`evaluate_title_author_against_isbn(..., filter_unreliable_gold=True)`
(default) now drops a gold pair when the gold work's author isn't
`names_compatible` with the query's own author, reporting the count as
`gold_quality_excluded` rather than silently dropping it. On the full
gold set (tuning fold): **9,223 of 23,697 gold pairs (39%) excluded** — the
ISBN-holdout eval's absolute recall/precision numbers going forward are
not comparable to any report persisted before this fix (`gold_pairs` and
`gold_quality_excluded` in the report make the population explicit; compare
deltas on same-methodology reports only). Spot-checked 25 excluded pairs
by hand: ~92% obviously-wrong gold (unrelated title/author or a stub);
~8% real matches OL only fails to link because the GR author is a
pseudonym `names_compatible` can't resolve (Jean Plaidy / Eleanor Burford)
— an acceptable, rare false-exclusion, and one the underlying matcher
couldn't have resolved either way (it uses the same `names_compatible`
gate).

**Acceptance-gate note:** one flagged reason — `suspicious_duplicate_targets`
9 vs baseline 8 (`docs` for the persisted-8 baseline: 2b-1's note above).
The new entry is an obscure self-published 4-book series ("Women on Top")
whose Goodreads titles annotate volume number in a free-text parenthetical
(`"(The Dud Wimpole Saga Book 1)"`) that `strip_series_suffix` doesn't
recognize (it expects `"(Series, #1)"`), so the volume-number guard above
has nothing to key off for this one series. `identity_recall`/`conflict`/
`false_merges` all moved sharply in the right direction and unit tests
pass. Judgment call: **accepted** — same reasoning pattern as 2b-1's
override, applied to a single low-`ratings_count` obscure-series edge case
rather than a systemic pattern. Backlog: teach `strip_series_suffix` (or a
sibling) to also recognize free-text `"Book N"` / `"Volume N"` annotations
outside the `(Series, #N)` parenthetical convention.

#### The curly-vs-straight apostrophe bug (2i)

Found while probing OL for real candidates during the Stage 2g draft-verdict
pass — not from inspection. `normalize_for_search`
(`tools/catalog/ol_common.py`, mirrored in
`Sources/SpineMatching/Normalization.swift`'s `normalizeForSearch`) grouped
curly single quotes (`'`/`'`, U+2018/U+2019) with the curly double-quotes as
"decorative punctuation" to strip, while deliberately *keeping* the straight
ASCII apostrophe (`'`, U+0027) — correct on its own (`test_keeps_meaningful_
marks` / `testStripsDecorativePunctuationButKeepsMeaningfulMarks`:
apostrophes distinguish "O'Brien" from "OBrien", so shouldn't be nuked like
real punctuation). The bug: curly and straight apostrophes are the same
character, glyph choice only, but got two different normalized outputs —
`normalize_for_search("Assassin's Blade")` (straight) → `"the assassin's
blade"`, `normalize_for_search("Assassin's Blade")` (curly) → `"the
assassins blade"`. Confirmed directly against `full.sqlite`: OL holds Sarah
J. Maas's *The Assassin's Blade* under the curly form
(`/works/OL17546674W`, 19 editions) — a Goodreads record for the same book
using the straight form never collided with it and landed on `unmatched`,
not because OL lacks the book.

Rough blast radius before the fix (upper bound — presence of an apostrophe
doesn't guarantee this was *why* a given row failed, but the asymmetry
itself was strictly a bug, never a legitimate tradeoff): of the pre-fix
6,300 `unmatched` / 1,972 `ambiguous` records, **601 / 218 respectively
contained an apostrophe character**.

Fix: fold both curly apostrophe glyphs to the straight form (not strip
either) in both language ports, so "apostrophes are meaningful" now holds
identically for both glyphs instead of only the straight one. New
regression tests in both languages
(`test_curly_apostrophe_folds_to_straight` /
`testCurlyApostropheFoldsToStraight`).

**Second-order finding, same stage:** the code fix alone didn't change
candidate retrieval yet. `full.sqlite`'s `books.titleNormalized` /
`authorNormalized` columns are *precomputed and stored at catalog-build
time*, not derived live — the SQL title-probe (`WHERE titleNormalized IN
(...)`, `match_goodreads.py`) kept reading the stale pre-fix values, so
"The Assassin's Blade" was still `unmatched` even after the code fix plus a
full re-run. Confirmed directly: `full.sqlite`'s stored `titleNormalized`
for OL17546674W was still `"the assassins blade"` (apostrophe stripped,
old behavior) after the code change. Fix: a targeted, derived-column-only
refresh, not a full catalog rebuild —

1. Backed up the pre-fix `titleNormalized`/`authorNormalized` for just the
   affected rows to a JSON file (row-level backup, not a full 21GB file
   copy — disk headroom was tight, ~36GB free on a 21.7GB database).
2. `UPDATE books SET titleNormalized = ?, authorNormalized = ? WHERE id =
   ?` over the **12,668 of 35.6M rows** whose title or author contains a
   curly apostrophe — every one of the 12,668 candidates actually needed
   the update (the asymmetry was total, not partial, for this character
   class).
3. `books_fts` (FTS5, `content='books'`) has triggers that re-index
   automatically on `UPDATE` to the content table, so no separate FTS
   rebuild step was required — verified with a direct exact-match lookup
   and an FTS sanity query afterward.

This only touches a *derived* index used for search, never the raw
OL title/author strings — the "raw data is irreplaceable, derived is
rebuildable" rule (`AGENTS.md`) applies as intended.

Measured effect (`--eval-isbn-holdout --fold tuning`, `full.sqlite`, code
fix + column refresh together vs. the 2h baseline):

| | 2h (before) | 2i (after) |
|---|---|---|
| `identity_recall` | 0.8025 | 0.8050 |
| `precision` | 0.9104 | 0.9108 |
| `conflict` | 1,297 | 1,292 |
| full-corpus `match_method` | `title_author` 14,201 / `ambiguous` 1,972 / `unmatched` 6,300 | `title_author` 14,234 (**+33**) / `ambiguous` 1,964 (**-8**) / `unmatched` 6,275 (**-25**) |
| `suspicious_duplicate_targets` | 9 (2h override, see above) | **8** — back at the pre-2h baseline |

Most of the full-corpus recovery (+25 of the +33 `title_author` gain) came
from the column refresh step, not the code fix in isolation — the
ISBN-holdout eval's own gold-pair lookup computes `title_core` live from
raw title/author text and never depended on the stale column, so it
under-measured the code fix's real impact on what actually ships: full-corpus
candidate retrieval via the SQL title-probe.

`acceptance_gate.py` vs the 2h baseline: **accepted, no override needed** —
every gated metric moved in the right direction or held.

**Backlog, not fixed here:** the same DB probing surfaced a second, smaller
title-normalization mismatch — GR and OL disagree on `"&"` vs `"and"`
within the same title in at least one observed pair (Carissa Broadbent,
GR's "The Serpent and the Wings of Night" vs OL's "The Serpent & the Wings
of Night," and the reverse direction for that series' book 2). Left open:
this needs a token substitution (`"&"` ↔ `"and"`), a different fix shape
than a single-character fold, with its own false-merge risk to check
(unlike the apostrophe fix, "and" is a real word that could appear
elsewhere in a title, not just as a conjunction standing in for "&").

#### Trustworthy evaluation: folds, an adversarial set, and persisted baselines

Tuning a matcher against the same fixed ISBN-gold set on every iteration
eventually fits noise in that set (Goodhart's law for record linkage). Three
additions make `--eval-isbn-holdout` safe to iterate against and give it a
negative-set check the ISBN-gold set — all positive pairs — cannot provide:

- **Tuning/validation fold** — `bibliographic_join.fold_for(book_id)` gives
  every `book_id` a fixed, reproducible ~80/20 split via a salted hash (no
  state to persist). Iterate against `--fold tuning`; touch `--fold
  validation` only at the start and end of a tuning run. A hash of `book_id`
  alone is already uncorrelated with popularity, so it doesn't skew toward
  the tail the way slicing the (popularity-ordered) gold set by position
  would — `bibliographic_join.fold_split_stats` is the empirical check that
  proves it.
- **Adversarial pairs** — [`tools/catalog/matcher_adversarial_pairs.yaml`](../tools/catalog/matcher_adversarial_pairs.yaml)
  is a small, hand-curated set of real book pairs that must never resolve to
  the same `workKey` (e.g. Meyer's vs. Koontz's *Twilight*). `--eval-isbn-holdout`
  folds `bibliographic_join.evaluate_adversarial_pairs`'s `false_merges`/
  `adversarial_total` into the same report by default (`--adversarial-pairs`
  to override, a nonexistent path to skip). Add a pair whenever a real false
  merge is found — this set is meant to grow, not stay static.
- **Persisted baseline** — `--eval-out auto` writes the report (plus git SHA
  and a UTC timestamp) to `catalog_goodreads('matcher_eval/<timestamp>_<short_sha>.json')`
  (or `--eval-out <path>` for an explicit location). Omitted by default —
  print-only, no disk writes:

```bash
python tools/catalog/match_goodreads.py --ol-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --book-show-api $BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl \
  --eval-isbn-holdout --fold tuning --eval-out auto
```

Two more checks run over the *full* `matched_goodreads.jsonl.gz` (the
ISBN-gold set only covers a minority of books) and need no ground truth —
[`tools/catalog/eval_canaries.py`](../tools/catalog/eval_canaries.py):
`suspicious_duplicate_targets` (distinct GR `book_id`s sharing a `workKey`
whose own title/author don't look like the same book — the strongest
available over-merge alarm on books with no ISBN to check against),
`gap_fill_candidates` (mirrors `gap_fill_unmatched`'s insert gate; should
fall as matching improves), and `match_method_distribution` (raw counts,
plus a delta against a prior `--baseline` report):

```bash
python tools/catalog/eval_canaries.py \
  --matched $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz \
  --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matcher_eval/canaries.json
```

#### Residual label set (Stage 2g): the one signal a human, not code, provides

Every check above has a blind spot: the ISBN-gold set only covers books
with a harvested ISBN that hits `book_isbns`; the adversarial set only
covers hand-picked known-distinct pairs, not real matcher output; the
canaries flag *patterns* (a shared `workKey`, a rising unmatched count)
without confirming any specific book is actually right or wrong. None of
them can say "book X really is/isn't OL work Y" for a book with no ISBN
that isn't already an adversarial pair. That gap is exactly the population
most likely to be silently wrong, and closing it needs a human to look at
the actual title/author, not another heuristic.

[`tools/catalog/sample_matcher_residual_candidates.py`](../tools/catalog/sample_matcher_residual_candidates.py)
picks *what* to review — never the answer — by sampling four strata from a
real `matched_goodreads.jsonl.gz` run, each one a case none of the checks
above can currently verify:

| `sample_reason` | What it means | Why it's worth a human look |
|---|---|---|
| `unmatched_popular` | `match_method="unmatched"`, high real `ratings_count` | No OL candidate was found at all — is OL actually missing the book, or did the title-probe miss it (e.g. a foreign-language edition title)? |
| `ambiguous_popular` | `match_method="ambiguous"`, high real `ratings_count` | The matcher found candidates but declined to pick — `candidate_*` shows its top-scored guess for reference only. |
| `low_margin_title_author` | `match_method="title_author"` with the smallest `match_margin` | Confident enough to pick, but the closest calls — the most likely `title_author` matches to be silently wrong. |
| `suspicious_duplicate` | one entry per `book_id` inside an `eval_canaries.suspicious_duplicate_targets` group | Multiple GR books share one `workKey` with metadata that doesn't obviously agree — asks whether *this* `book_id` really belongs there. |

```bash
python tools/catalog/sample_matcher_residual_candidates.py \
  --matched $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz \
  --ol-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --out tools/catalog/matcher_residual_labels.yaml
```

Every generated entry has `verdict: null`. A human reviews each one and
sets `verdict` to `correct` (the shown `candidate_work_key` really is this
book), `wrong` (it isn't — fill in `corrected_work_key` if the right OL
`workKey` is known), `no_ol_match` (this GR book genuinely has no OL entry),
or `unsure`. `bibliographic_join.load_residual_labels` skips any entry
without a `verdict` — unreviewed rows are never mistaken for ground truth —
and `evaluate_residual_labels` scores the matcher's *current* predictions
against the reviewed subset, reporting `accuracy` and a `mismatches` list.
Re-running the sampler overwrites `--out` with a fresh, unverified sample;
copy any verdicts worth keeping (or diff against git) before re-sampling.

This set is small (100-200 books) and expensive per label (a human, not a
script) — it is not a substitute for the checks above, and it should not be
tuned against repeatedly. Its purpose in Phase 4 (the bounded multi-agent
tuning loop) is a final, trustworthy check on exactly the population the
automated metrics can't reach — not a training signal to iterate against
every pass, which would just move the Goodhart's-law problem here instead
of solving it.

Linkage (does a book attach to the right OL work?) is necessary but not
sufficient — the shipped 50k slice is decided by *rank*
(`compute_shelf_score` / `rerank_popularity`).
[`tools/catalog/eval_popularity_ranking.py`](../tools/catalog/eval_popularity_ranking.py)
checks that separately: Spearman correlation and top-1k/10k/50k overlap
between `shelf_score`'s rank and `book_show_api`'s own `ratings_count` (via
`consolidated_signals.jsonl.gz` — the strongest available reference signal,
covering ~49k books today), same `--fold` discipline as above:

```bash
python tools/catalog/eval_popularity_ranking.py \
  --matched $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz \
  --consolidated-signals $BOOK_SPINES_DATA/derived/book-catalog/goodreads/consolidated_signals.jsonl.gz \
  --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matcher_eval/ranking.json
```

**This eval measures agreement with Goodreads, not shelf presence.** Both
`shelf_score` and its reference signal (`book_show_api`'s own
`ratings_count`, via `consolidated_signals.jsonl.gz`) are Goodreads
engagement metrics. There is no ground-truth physical-ownership signal
anywhere in this pipeline — "commonly on a physical bookshelf" is not
directly measured, only proxied by online rating/list activity. Raising
Spearman or top-k overlap against this reference makes `shelf_score` a
*better copy of Goodreads popularity*, which is the right thing to optimize
given the data actually available, but a win here does not by itself prove
improved shelf-prediction. Known blind spots in the proxy, neither
measured nor corrected anywhere in this pipeline:

- **Backlist / classics / reference / gift books** are plausibly
  under-rated online relative to how often they sit on a physical shelf —
  bought once, kept for decades, rarely re-reviewed (e.g. dictionaries,
  older literary classics, coffee-table/gift editions).
- **Digital-first / BookTok-era titles** are plausibly over-rated online
  relative to physical ownership — heavy online engagement doesn't always
  convert to a printed copy on a shelf.

Treat "beats the ranking eval" as "closer to Goodreads' own popularity
signal," not as "proven correct" for the actual downstream goal (OCR
shortlist accuracy on real bookshelves). Revisit only if a genuinely
independent signal becomes available (e.g. library holdings data) — no
such signal exists in this pipeline today, and none is planned.

[`tools/catalog/acceptance_gate.py`](../tools/catalog/acceptance_gate.py) is
the reusable accept/reject check a tuning loop runs against two of these
reports: accept only if, versus baseline, `identity_recall` doesn't
decrease, and `conflict`/`false_merges`/`suspicious_duplicate_targets` don't
increase, and unit tests still pass. Deliberately asymmetric — a false
merge is worse than a miss for this pipeline (a miss still gets a synthetic
gap-fill row with the *correct* title; a false merge transfers popularity
onto the *wrong* one) — see the module docstring for the full rationale:

```bash
python tools/catalog/acceptance_gate.py --baseline matcher_eval/<baseline>.json --candidate matcher_eval/<candidate>.json
```

#### Genre tags (list-derived, not official Goodreads `bookGenres`)

Every matched row in `matched_goodreads.jsonl.gz` carries a `genres` field — the union of every seed list's
`genre` tag a book appeared under (e.g. a book on both the fantasy and romance seed lists gets
`"genres": ["fantasy", "romance"]`). **These are tags on the Listopia *lists* the book happened to be scraped
from, not Goodreads' own per-book category system** (`bookGenres`, sourced from user bookshelves and only
available via a separate, not-yet-built `book_show` scrape of individual book pages — see the module docstrings
in `tools/scrape_goodreads_lists.py` / `tools/catalog/match_goodreads.py` for that distinction). Roughly ~19% of
matched books carry more than one tag simply because they appear on more than one seed list.

`tools/catalog/match_goodreads.py` also writes `genre_tags.json` next to `matched_goodreads.jsonl.gz` on every
run — a tag -> display-name lookup built from `goodreads_seed_lists.yaml`'s optional `short_label`/`list_title`
fields (falling back to a title-cased tag / de-underscored slug when omitted), so a `genres` tag like `"fantasy"`
can be rendered as "Fantasy" or "Best Fantasy Books" without hardcoding a mapping elsewhere:

```json
{
  "fantasy": {
    "short_label": "Fantasy",
    "list_title": "Best Fantasy Books",
    "slug": "Best_Fantasy_Books",
    "list_id": 367
  }
}
```

If two seed lists ever share the same `genre` tag, the first one (in seed-yaml order) wins the mapping entry and
a warning is printed — every tag in the seed yaml is unique today, but nothing enforces that as the list grows.

#### Curating Goodreads seed lists

[`tools/catalog/goodreads_seed_lists.yaml`](../tools/catalog/goodreads_seed_lists.yaml) is hand-curated and never
rewritten programmatically. As it grows (by hand, or via the `list_discovery` harness profile), some lists end up
being near-duplicates of each other (e.g. "Best Science Fiction" and "Best Science Fiction & Fantasy") — scraping
both wastes session budget and inflates `list_appearances`/`shelf_score` for books that just happen to be on
several synonym lists.

[`tools/catalog/analyze_goodreads_lists.py`](../tools/catalog/analyze_goodreads_lists.py) answers "which lists are
redundant?" from already-scraped data — pairwise Jaccard overlap, containment (is list A basically a subset of
list B?), top-N popularity coverage, a greedy set-cover order, and how many *new* books each list adds to a
top-5000-by-`ratings_count` corpus beyond every other list combined:

```bash
# Report only -- always safe, never writes anything but the report itself
python tools/catalog/analyze_goodreads_lists.py
python tools/catalog/analyze_goodreads_lists.py --top-n 1000   # default is 1000

# Review the report, then opt in to writing deprecation entries
python tools/catalog/analyze_goodreads_lists.py --apply-deprecations
```

Only lists marked `done` in the scrape checkpoint are evaluated — a list still `pending`/`error` hasn't finished
scraping, so there's nothing meaningful to say about its overlap yet; it just shows up under `not_yet_evaluated`
in the report. A two-outcome rule engine (`keep` or `deprecate`) runs per `done` list:

| Rule | Condition | Outcome |
|---|---|---|
| Anchor exemption | `list_type: anchor` in the seed yaml | always `keep` (never evaluated further) |
| Subset | >70% of this list's books already appear in one other single `done` list | `deprecate` |
| Overlap + low value | Jaccard > 0.35 with another `done` list **and** it adds < 50 new books to the top-5000 corpus | `deprecate` |
| Default | none of the above | `keep` |

`list_type` is an optional field on each seed entry (defaults to `genre` when omitted); mark exactly one broad
popularity list (e.g. "Best Books Ever") as `list_type: anchor` so it's never flagged no matter how much it
overlaps with everything else — that's expected for a list spanning every genre.

The report is written to `list_recommendations.json` next to the raw scrape data. `--apply-deprecations` only ever
*adds* new `deprecate` entries to a separate `seed_list_overrides.yaml` sidecar (generated under
`catalog_goodreads()`, not committed, and never written to `goodreads_seed_lists.yaml` itself) — it never
overwrites an entry you've already hand-edited, so manually overriding a rule's call (e.g. setting
`curation_status: active` to keep a list anyway) sticks across future runs:

```yaml
overrides:
  3:
    curation_status: deprecated
    reason: "subset: 87% overlap with list 19341 (scifi)"
    set_at: "2026-07-28T20:00:00Z"
```

`curation_status: deprecated` only controls **future scraping budget** for that list — it does not delete or
exclude already-scraped raw data, which keeps contributing to `match_goodreads.py`'s signals regardless. To
actually skip deprecated lists during a scrape session, opt in on `scrape_goodreads_lists.py`:

```bash
python tools/scrape_goodreads_lists.py --skip-deprecated
```

Omitting `--skip-deprecated`, or having no `seed_list_overrides.yaml` yet, behaves exactly as before this tool
existed — both are strictly opt-in.

### scrape-harness dependency

The Goodreads side of this pipeline is a **consumer of a separate, generic tool** —
[`scrape-harness`](https://github.com) (sibling checkout, default `~/dev/scrape-harness`; see
`tools/scrape_goodreads_lists.py`'s `--harness-root`). `scrape-harness` knows nothing about
`book-train`; it just runs a YAML-described scraping profile and writes JSONL. That split is
correct, but it means the JSONL **field contract** between the two repos is easy to break silently
by editing a profile without updating this side (see `scrape-harness/README.md`'s own "Consumers"
section for the same contract documented from the tool's point of view).

Profiles used, all under `scrape-harness/sites/goodreads/profiles/`:

| Profile | Invoked by | Output |
|---|---|---|
| `list_show` | `tools/scrape_goodreads_lists.py`, one subprocess call per seed list | `catalog_goodreads('raw')/<list_id>.jsonl` |
| `book_show_api` | manual batch run (see below) — a Next.js JSON API lane, not a browser page render | `catalog_goodreads('book_show_api.jsonl')` |
| `book_show` | optional, older per-book HTML scrape (superseded by `book_show_api` for new work) | `catalog_goodreads('book_show')/<book_id>.jsonl` |

`list_show` paginates via direct `?page=N` URLs (not next-button clicks) — see
[`list_show.yaml`](https://github.com) and `scrape_harness.runtime.run_profile`. A resumed run
does a single `goto` straight to the next unsaved page instead of clicking through every
already-saved page, and the runtime detects a genuinely-exhausted list two ways: an extracted page
with zero `book_urls`, or (weaker signal) a `wait_for` timeout past page 1 — confirmed live that
Goodreads renders the list shell but hides `table.tableList` entirely once `page` exceeds a list's
real last page, so a naive `wait_for(state=visible)` would otherwise time out instead of failing
fast. `tools/scrape_goodreads_lists.py` additionally chunks each harness invocation to
`--max-pages-per-run` (default 50, via a generated `list_show_chunked` profile copy in the
harness's `sites/goodreads/profiles/`) so a long list (e.g. list 1's ~790 pages) runs as ~16 short
sessions instead of one multi-hour process; a chunk boundary reports `chunked` (scheduled like
`pending`, not a failure) rather than `done`. `error`/`empty_unexpected` results get an increasing
per-list retry cooldown (`Checkpoint.next_retry_at`, enforced across separate CLI invocations), and
`SOFT_BLOCK_THRESHOLD` (3) consecutive plain-timeout errors on one list are treated like a
`challenged` result — session stops early even without a textual challenge marker.

`match_goodreads.py`'s `_LIST_SHOW_FIELDS` (validated at the top of every run by
`validate_list_show_schema` — a broken contract raises `SchemaError` immediately rather than
silently producing null ratings/failed matches) requires every `list_show` record to carry:

| Field | Read by | Used for |
|---|---|---|
| `book_urls` | `parse_book_id` | Goodreads book ID |
| `titles` | `parse_list_show_file` | title+author bibliographic join vs. Open Library |
| `authors` | `parse_list_show_file` | title+author bibliographic join vs. Open Library |
| `rating_texts` | `parse_rating_text` | `avg_rating`, `ratings_count` |
| `score_texts` | `parse_score_text` | `list_score_sum` (shelf_score input) |
| `vote_texts` | `parse_vote_text` | `vote_sum` (shelf_score input) |

**If a `scrape-harness` change to `list_show.yaml` touches any of these fields, update
`_LIST_SHOW_FIELDS`/the parsers in `match_goodreads.py` in the same change** — don't rely on
`validate_list_show_schema` catching it at the next run; by then a re-scrape may already be needed.

`book_show_api` batch output is a separate, optional ISBN *overlay* on the title+author join
(exact ISBN is authoritative when it hits `book_isbns`; otherwise title+author stands)
read via `--book-show-api`, using direct `legacy_id`/`isbn13` fields (no blob parsing, unlike the
older `book_show`/`--book-show-dir` path). It's resumable/incremental —
[`tools/catalog/extract_remaining_ids.py`](../tools/catalog/extract_remaining_ids.py) computes which
book IDs from `raw/*.jsonl` still need fetching (maintaining a `fetched_ids.txt` sidecar so re-runs
don't re-parse every `book_show_api*.jsonl` shard) and writes `ids_remaining.txt`.

Some books genuinely have no ISBN13 in Goodreads' data, so `book_show_api` can never produce a
`legacy_id`/`isbn13` for them — every attempt comes back as an `incomplete_record` warning (see
below). `extract_remaining_ids.py` tracks these per-book: once a book accumulates
`--give-up-after` (default 3) warning-only attempts without ever succeeding, it's excluded from
`ids_remaining.txt` and recorded in a second sidecar, `book_show_api_gave_up.txt`, so the batch
scrape stops retrying it forever. This is permanent — a book only comes back into
`ids_remaining.txt` if you delete its id (or the whole file) from `book_show_api_gave_up.txt` to
force a retry. A surprisingly large gave-up count in the summary line likely means an extraction
bug rather than that many books genuinely lacking ISBNs, and is worth spot-checking.

Within each of the never-tried/retry buckets, `ids_remaining.txt` is ordered by popularity —
highest `ratings_count` first, then `list_appearances`, then `list_score_sum`, with `book_id` as a
final deterministic tie-break — so the scrape fetches the books most likely to matter first instead
of an arbitrary (or, previously, randomly shuffled) order. Never-tried IDs stay ahead of retries.
A book whose warning-only attempt count just increased is withheld for `--retry-cooldown-hours`
(default 24) via `book_show_api_retry_after.json` so the next chunk does not immediately re-request
it. The signal comes entirely from the already-scraped `list_show` raw data (no new scrape, no OL
lookup) via [`tools/catalog/list_show_popularity.py`](../tools/catalog/list_show_popularity.py), the
same module `analyze_goodreads_lists.py` uses for its own overlap/coverage report. A
`book_popularity.json` sidecar caches the aggregation, rebuilt only when it's missing or older than
the newest `raw/*.jsonl` — the same staleness pattern as `fetched_ids.txt`. Pass `--order shuffle`
to fall back to the old random never-tried ordering; that's a debugging escape hatch only, not
expected to be needed day-to-day.

`scrape-harness`'s API-mode runtime appends to `--out` rather than rewriting it, so restarting a
`book_show_api` scrape can never truncate records a previous session already wrote. In-session pacing
is consecutive-failure AIMD plus a type-aware circuit breaker (`_AdaptiveDelay` + `_SoftFailureCircuit`
in `scrape_harness.api_runtime`):

- Isolated `incomplete_record` (200, missing ISBN) does not change delay; two or more consecutive
  incompletes multiply delay by 1.5. The cap keeps the baseline min/max ratio (3000/8000 →
  11250–30000ms), not a collapsed 30s–30s sit. Hitting the cap, or a rolling incomplete rate of
  ≥25% over ≥20 samples (window 40), stops the session with **exit 3**.
- **404 / `json_parse_error` / bot-wall do not enter AIMD.** Isolated 404s skip the book at the
  current delay. 8 consecutive 404s stop the session (stale Next.js build id). 3 consecutive
  `blocked_suspected` stop. 12 consecutive `incomplete_record` also stop. Chrome is not recycled
  mid-session for ordinary incompletes. The API client does not chain `Referer` from one
  `_next/data` URL to the next.

The easiest way to drive all of this unattended for a large `ids_remaining.txt` is the orchestrator loop:

```bash
cd ~/dev/book-train
# Container on the Mac: scrape traffic through Mullvad (Gluetun). Host stays off-VPN.
# Human first: Docker running, quit Mac Mullvad + Hotspot Shield, new WireGuard
# device in tools/scrape-container/.env, `scrape-harness vpn baseline` with VPN off.
# One-time: disk-cleanup nas-copy/gdrive-sync --item ml-goodreads-scrape --execute
# (see DATA.md), then:
.venv/bin/python tools/run_book_show_api_loop_detached.py prepare
.venv/bin/python tools/run_book_show_api_loop_detached.py start
.venv/bin/python tools/run_book_show_api_loop_detached.py status
```

Native Mac (Mullvad app connected, host not using the Docker VPN):
`.venv/bin/python tools/run_book_show_api_loop_detached.py start-native` — do not attach `caffeinate -i tools/run_book_show_api_loop.sh` to an Agent shell.

Headed Chrome in the container needs Xvfb on `:99`. [`tools/catalog/ensure_xvfb.py`](../tools/catalog/ensure_xvfb.py) starts it and, after a container restart or Xvfb crash, clears a stale `/tmp/.X99-lock` and restarts. The entrypoint and each loop iteration call `ensure` before discover. A Chrome `Missing X server` / no-XServer failure is a **display** problem: the loop retries `ensure` once and then **stops** — it does not rotate Mullvad. Native Mac skips this (Aqua is the display). Babysit with `ensure_xvfb.py status` or the `[display]` lines in compose logs.

Each iteration: refresh `ids_remaining.txt` -> `scrape-harness doctor` (VPN preflight; on failure,
sleeps 15 minutes and retries rather than aborting) -> **prepare-exit** (wipe
`book_show_api_chunked`, visit Goodreads home, open Best Fantasy Books — the list that
contains the discover target — then click through to book 33 / fall back to the book URL,
extract the Next.js build id in that same Chrome profile; navigations use a 400–1200ms
fast-user pause and a 15s page timeout, not a 15-minute wait) -> one
`book_show_api_chunked` scrape **in that same profile** (a generated copy of
`book_show_api.yaml` capped at `policy.max_requests: 120` — see
[`tools/catalog/write_book_show_api_chunked_profile.py`](../tools/catalog/write_book_show_api_chunked_profile.py);
do not pass `--fresh-browser` here) -> optional Mullvad city rotate (`MULLVAD_ROTATE=1`) ->
a cooldown sleep -> repeat until nothing remains.

If prepare-exit / discover fails, the loop rotates to the next Mullvad city immediately and
retries with a new fresh profile. After one full walk of the city pool it sleeps 15 minutes
once; a second full walk hard-stops (`discover_fail_state.json`) so a bad HTML edge cannot
spin. Implemented in
[`tools/catalog/book_show_api_exit_session.py`](../tools/catalog/book_show_api_exit_session.py)
and scrape-harness `api discover --warmup-url` / `--user-data-id`.
`HARNESS_ROOT`, `MAX_REQUESTS` (default 120),
`COOLDOWN_SECONDS`, `DOCTOR_RETRY_SECONDS`, `SPIKE_MAX_TIERS`, and `MULLVAD_ROTATE` / `MULLVAD_CITIES`
env vars override the defaults. A harness **exit 3** (delay cap or rolling incomplete rate) skips
the 80%-spike `check` gate, runs `escalate-cooldown` on the same 15m → 45m → 2h ladder, and does
**not** rotate Mullvad. Three consecutive controlled stops hard-stop the loop.

Doctor requires a matching VPN **process** (`mullvad-daemon` / `Mullvad VPN` / `gluetun`) **or**
`VPN_STATUS_URL` returning `{"status":"running"}`, plus an exit IP that is not the recorded home
baseline (`scrape-harness vpn baseline`, VPN off, on the Mac). The container sets
`VPN_STATUS_URL=http://127.0.0.1:8000/v1/vpn/status` on the Gluetun control API (not published
on the host). Record the baseline on the Mac with Mullvad quit.

Discover-fail always rotates the Mullvad city (not gated on `MULLVAD_ROTATE`). The container
defaults `MULLVAD_ROTATE=1` so [`tools/catalog/mullvad_rotate.py`](../tools/catalog/mullvad_rotate.py)
also walks eight US cities during the **healthy 60s inter-chunk pause**:
`nyc` `lax` `chi` `dal` `sea` `atl` `mia` `qas`.
In the container set `MULLVAD_BACKEND=gluetun`. Override the pool with `MULLVAD_CITIES="nyc lax chi"`.
Rotations append to `mullvad_rotations.jsonl`.

Each chunk also updates a timestamped progress sidecar via
[`tools/catalog/book_show_api_progress.py`](../tools/catalog/book_show_api_progress.py)
(`mark-start` before the scrape, `record` after). `book_show_api.jsonl` has no per-row
timestamps; the sidecar is how “how many ISBN scrapes in the last 24h?” stays exact:

```bash
.venv/bin/python tools/catalog/book_show_api_progress.py --hours 24
.venv/bin/python tools/catalog/book_show_api_progress.py summarize --hours 24 --json
```

After each chunk, [`tools/catalog/book_show_api_session_health.py`](../tools/catalog/book_show_api_session_health.py)
`check`s the new JSONL rows. A **failure spike** is ≥80% warnings with ≥20 records, **or** ≥8 trailing
warnings (the harness circuit-break). It is classified and recovered instead of exiting whenever the
catalog looks fine:

| Kind | Signal | Recovery |
|---|---|---|
| `catalog` | empty `book_popularity.json` / mis-ordered `ids_remaining.txt` | rebuild queue, 5-minute cooldown |
| `stale_build` | majority `json_parse_error` or HTTP 404, including a trailing 404 run after a healthy prefix | wipe the chunked Chrome profile, warmup home → list → book 33 `api discover` (must GET `_next/data/{id}/book/show/33.json` as 200 JSON before writing `next_build.yaml`), probe book 33 in that same profile; if the probe fails, retry discover after 60s then 180s before `soft_block` |
| `soft_block` | `incomplete_record` (the usual IP throttle) | wipe the chunked Chrome profile, refresh the Next.js build id (same warmup + shared profile), rotate Mullvad if `MULLVAD_ROTATE=1`, probe a known-good book, escalate cooldown **15m → 45m → 2h** (capped at 60–180s after a successful rotate) |
| `hard_block` | majority `blocked_suspected` | rotate Mullvad if enabled and continue; otherwise exit immediately |

A healthy chunk resets `spike_recovery_state.json`. After any recovery the next chunk drops `DELAY_MS_*`
and uses the 3000–8000ms baseline. After three failed soft-block probes the loop hard-stops — re-run
`.venv/bin/python tools/run_book_show_api_loop_detached.py start-native` (safe to resume; JSONL is append-only).

To run one session by hand instead (e.g. for a small delta, or to debug a profile change):

```bash
cd ~/dev/book-train
python tools/catalog/extract_remaining_ids.py   # -> ids_remaining.txt

cd ~/dev/scrape-harness
uv run scrape-harness scrape goodreads --profile book_show_api --fresh-browser \
  --set-from-file book_id=$BOOK_SPINES_DATA/derived/book-catalog/goodreads/ids_remaining.txt \
  --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl

cd ~/dev/book-train
python tools/catalog/match_goodreads.py --ol-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --book-show-api $BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl \
  --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz
```

Safe to re-run `extract_remaining_ids.py` at any point — before all seed lists finish scraping, after
adding a new list, or to resume a `book_show_api` run that was interrupted; it only ever asks for the
delta not already fetched. If a `book_show_api` batch ever does end up split across more than one JSONL
(e.g. a manual `--out` under a different name), dedupe-merge them into one canonical file with
[`tools/catalog/merge_book_show_api.py`](../tools/catalog/merge_book_show_api.py) before passing
`--book-show-api` to `match_goodreads.py`, which only reads a single file. The merge preserves each
warning's `_attempt_count` (summing repeats of the same book across shards) so give-up tracking above
stays correct afterward:

```bash
python tools/catalog/merge_book_show_api.py --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl
# (no arguments: merges every catalog_goodreads('book_show_api*.jsonl') shard in place)
```

All of this data lives under `catalog_goodreads()` (`tools/paths.py`) —
`$BOOK_SPINES_DATA/derived/book-catalog/goodreads/` — outside both repos, not committed to either.

**Consolidating every popularity signal already on disk.** `list_show` (ratings/list-membership) and
`book_show_api` (ISBN + ratings/genres) are scraped independently, and `book_show_api` itself has two
outcomes: a full success (`legacy_id` present) and an `_scrape_warning: incomplete_record` row (no
`isbn13`/`legacy_id`/`title`, but the ratings/genres fields usually still came through — previously
unused). [`tools/catalog/consolidate_popularity_signals.py`](../tools/catalog/consolidate_popularity_signals.py)
merges all three into one per-`book_id` table, recovering `incomplete_record` rows' `book_id` from
their `_url` field:

```bash
python tools/catalog/consolidate_popularity_signals.py \
  --book-show-api $BOOK_SPINES_DATA/derived/book-catalog/goodreads/book_show_api.jsonl \
  --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/consolidated_signals.jsonl.gz
```

Pass the output to `match_goodreads.py --consolidated-signals <path>` to fill `avg_rating`/
`ratings_count` gaps in `compute_shelf_score` from the `book_show_api` harvest — additive only, and off
by default: it never overrides a book's real `list_show` rating/ratings-count, so `matched_goodreads.jsonl.gz`
is unchanged unless this flag is passed.

**Pacing benchmark:** before lowering `book_show_api`'s delay range below the 3000-8000ms baseline,
prove it holds up on a fixed sample rather than guessing from a live-run streak that could equally be
explained by queue ordering or browser-profile state (see git history for that exact false alarm).
[`tools/catalog/bench_book_show_api_pacing.py`](../tools/catalog/bench_book_show_api_pacing.py) picks
a deterministic sample (default 20, `--seed 42`) of book IDs already proven fetchable (`legacy_id` +
`isbn13` both present in `book_show_api.jsonl`), runs it once at baseline and once at a faster tier
(1500-4000ms) through a generated `book_show_api_chunked` profile, and prints a side-by-side success
rate / bot-wall-hit report — writing only under `/tmp`, never touching the canonical
`book_show_api.jsonl`. It refuses to run while the live loop holds scrape-harness's `goodreads` site
lock. Bench first, then set `DELAY_MS_MIN`/`DELAY_MS_MAX` on the loop:

```bash
# stop the live loop first, then:
python tools/catalog/bench_book_show_api_pacing.py

# only if that exits 0 ("PROMOTE"):
DELAY_MS_MIN=1500 DELAY_MS_MAX=4000 caffeinate -i tools/run_book_show_api_loop.sh
```

## Schema (v1 + metadata)

Uses existing `books` + `books_fts` tables (`SpineCatalog` v1 migration). OL builds add
optional `popularityRank` and `editionCount` columns (v2 migration) for subset rebuilds.
One row per **work**; `workKey` is the OL work id (e.g. `/works/OL123W`).

**Co-authors:** `author`/`authorNormalized` include every credited author (OL order, first =
primary), not just `authors[0]` — see [`ol_common.py`'s `author_keys_from_work`/`join_author_names`](../tools/catalog/ol_common.py).

**Match-field dedup:** OL has a data-quality pattern where two distinct `workKey`s end up with
identical normalized `(title, author)` — ~9% of rows in a pre-dedup `full.sqlite`. Since each has
its own `workKey`, `AcceptPolicy`'s work-level dedup doesn't collapse them, and they wasted
`ios_en`/subset-cap slots and shortlist slots. `CatalogOLBuild.buildFromIntermediate` now dedupes by
normalized match field at build time (keeping the better-ranked row — `works.jsonl.gz` is already
popularity-ordered, so the first occurrence in the stream wins), so `full.sqlite` and every subset
profile derived from it (`ios_en`/`ios_en_shelf`/`dev_smoke`) ship deduped for free.
[`BookCatalog.dedupeByMatchFields`](../Sources/SpineCatalog/BookCatalogMatchFieldDedup.swift) is
the runtime counterpart — defense-in-depth for catalogs that don't go through this OL ETL (CSV
imports via `ol_to_csv.py`, hand-built/test catalogs, a pre-rebuild catalog still on a device),
applied in `retrieveRoleAware`/`retrieveCandidates` before any shortlist cap.

## book-id-ios

Bundle `ios_en.sqlite` under `Sources/BookID/Resources/` (optional resource — build with
`--install-ios`). `CatalogStore` copies it to Application Support on first launch. On macOS,
falls back to `$BOOK_SPINES_DATA/derived/book-catalog/ios_en.sqlite` when the bundle is absent.

## CLIs

```bash
# CSV import
swift run -c release catalog-build catalog.csv --db catalog.sqlite

# OL intermediate → SQLite
swift run -c release catalog-build --intermediate derived/book-catalog/intermediate \
  --output derived/book-catalog/ios_en.sqlite --languages eng --max-works 250000 --min-editions 2

# Subset from full
swift run -c release catalog-build --subset-from derived/book-catalog/full.sqlite \
  --output derived/book-catalog/ios_en.sqlite --languages eng --max-works 250000

# Full pipeline match test
swift run -c release spine-id <image> --db catalog.sqlite [--fm]
```

## Build pipeline performance

`tools/catalog/process_ol.py` and `CatalogOLBuild.buildFromIntermediate` are both dominated by
per-row Python-level query/insert overhead, not I/O, at `full.sqlite` scale (~36-39M works). All
four fixes below keep the existing SQLite-staging memory-safety model (bounded batches / key-count
accumulators, never a full in-memory dict of the whole corpus) — they just stop paying interpreter
overhead for what SQLite (or SQLite's own bulk-load idioms) can do as one set-based operation.

- **D1 — `ingest_works_raw` + `build_work_author_names` + `build_works_out` (SQL-join rewrite).**
  The old `ingest_works` did 3 sequential single-row `SELECT`s per work (author name, edition
  count, min ISBN) via individual `conn.execute()` calls — ~100M+ Python↔SQLite round trips across
  ~36-39M works. Replaced with: a bulk streaming stage into `works_raw`/`work_authors` (parsing
  only), one bulk ordered join + streaming groupby to pre-join co-author names into
  `work_author_names` (see co-author note above), then one set-based `INSERT ... SELECT ... JOIN`
  against `edition_stats`/`work_isbns` to populate `works_out`.
- **D2 — `executemany()` batching.** `ingest_editions` and `export_intermediate`'s per-`work_key`
  language/ISBN lookups both called `conn.execute()`/ran a subquery once per row inside a Python
  loop. `ingest_editions` now buffers rows per `BATCH` and writes with `executemany()`;
  `export_intermediate`'s per-work language/ISBN lookups are now two bulk `GROUP_CONCAT` queries
  instead of a subquery per work.
- **D3 — bulk-load pragmas on the Swift output DB.** `buildFromIntermediate`/`buildFromSubset` set
  `PRAGMA synchronous=OFF; PRAGMA journal_mode=MEMORY` for the bulk-insert phase (safe: the build is
  fully discardable on failure until it's done) and restore durable settings
  (`synchronous=FULL; journal_mode=DELETE`) in `finalizeCatalog`.
- **D4 — defer `books_fts` sync + secondary indexes.** `BookCatalog`'s migrator creates 4 secondary
  indexes on `books` and 3 FTS5 sync triggers before any row lands, so a from-scratch bulk insert
  paid incremental B-tree maintenance + per-row trigram tokenization ~35M times.
  `buildFromIntermediate` now captures the migrator's own `CREATE INDEX`/`CREATE TRIGGER` SQL from
  `sqlite_master` (not a hardcoded copy — can't drift from the migrator), drops those objects,
  bulk-inserts, then replays the captured SQL and runs `INSERT INTO books_fts(books_fts)
  VALUES('rebuild')` once (FTS5's own documented bulk-load idiom) before `finalizeCatalog`. Only
  `buildFromIntermediate` (the `full`-scale from-scratch OL build) does this — `buildFromSubset`'s
  smaller derived builds keep the normal always-indexed migrator path, which is already fine at
  that scale.

### Considered, not planned

Two ideas from the same performance review, lower confidence-of-payoff relative to their
implementation cost — not scheduled as work, written down so they aren't re-investigated from
scratch:

- **Pipelined parallelism for JSON decode/normalize.** Decompression + `json.loads` is CPU-bound
  and embarrassingly parallel per line, but the by-`work_key` aggregation itself doesn't parallelize
  cleanly without a real shuffle/partition step. Worth revisiting only if D1-D3 are measured and
  decode cost (not query pattern / bulk-load cost) turns out to dominate.
- **External sort-merge instead of SQLite staging.** Emitting `(work_key, ...)` rows as flat files
  sorted by `work_key` and doing a linear merge-join instead of SQLite joins/indexes trades random
  access for sequential I/O — the same underlying idea as D1, but as new bespoke merge/dedup/tie-
  break code instead of a SQL query the `sqlite3` CLI can reproduce ad hoc for debugging. Only worth
  it if D1's SQL-join rewrite is measured and found insufficient.

## Licensing

Open Library data is [CC0](https://openlibrary.org/developers/dumps). No cover images shipped.

## Tests

```bash
python tools/catalog/test_ol_common.py
python tools/catalog/test_process_ol.py
python tools/test_scrape_goodreads_lists.py
python tools/catalog/test_list_show_popularity.py
python tools/catalog/test_consolidate_popularity_signals.py
python tools/catalog/test_analyze_goodreads_lists.py
python tools/catalog/test_bibliographic_join.py
python tools/catalog/test_match_goodreads.py
python tools/catalog/test_eval_popularity_ranking.py
python tools/catalog/test_eval_canaries.py
python tools/catalog/test_acceptance_gate.py
python tools/catalog/test_extract_remaining_ids.py
python tools/catalog/test_merge_book_show_api.py
python tools/catalog/test_write_book_show_api_chunked_profile.py
python tools/catalog/test_book_show_api_progress.py
python tools/catalog/test_book_show_api_session_health.py
python tools/catalog/test_mullvad_rotate.py
python tools/catalog/test_container_scrape.py
python tools/test_run_book_show_api_loop_detached.py
python tools/catalog/test_bench_book_show_api_pacing.py
swift test --filter BookCatalogTests
swift test --filter BookCatalogISBNAndRoleRetrievalTests
swift test --filter CLIIntegrationTests
```
