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
swift test --filter BookCatalogTests
swift test --filter CLIIntegrationTests
```
