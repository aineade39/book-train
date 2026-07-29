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

# Match scraped books against full.sqlite
python tools/catalog/match_goodreads.py --ol-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --out $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz

# Rebuild ios_en_shelf.sqlite from a scratch copy of full.sqlite (never mutates full.sqlite itself)
python tools/catalog/build_ios_en_from_goodreads.py \
  --full-db $BOOK_SPINES_DATA/derived/book-catalog/full.sqlite \
  --intermediate-dir $BOOK_SPINES_DATA/derived/book-catalog/intermediate \
  --matched-goodreads $BOOK_SPINES_DATA/derived/book-catalog/goodreads/matched_goodreads.jsonl.gz \
  --output $BOOK_SPINES_DATA/derived/book-catalog/ios_en_shelf.sqlite
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

## Schema (v1 + metadata)

Uses existing `books` + `books_fts` tables (`SpineCatalog` v1 migration). OL builds add
optional `popularityRank` and `editionCount` columns (v2 migration) for subset rebuilds.
One row per **work**; `workKey` is the OL work id (e.g. `/works/OL123W`).

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

## Licensing

Open Library data is [CC0](https://openlibrary.org/developers/dumps). No cover images shipped.

## Tests

```bash
python tools/catalog/test_ol_common.py
python tools/test_scrape_goodreads_lists.py
python tools/catalog/test_analyze_goodreads_lists.py
python tools/catalog/test_match_goodreads.py
swift test --filter BookCatalogTests
swift test --filter CLIIntegrationTests
```
