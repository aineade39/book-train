# OCR-to-catalog fuzzy match: algorithm comparison (bookcase.jpg)

Generated 2026-08-17 from `tools/experiment_match_algorithms.py` against oracle
ground truth in `optimize-gemini/fixtures/oracles/bookcase.json`.

## Setup

| Item | Value |
|---|---|
| Scene | `bookcase.jpg` |
| Oracle-paired spines (quality gate passed) | 89 |
| ios_en run | `CFBA76E6-53F8-4F07-A8CF-051326ABA660` (catalog match ~4.8 s) |
| full run | `A5C553DE-4345-40D2-A085-CD2662FACFA1` (catalog match ~199 s) |
| ios_en DB | `$BOOK_SPINES_DATA/derived/book-catalog/ios_en.sqlite` (250k rows) |
| full DB | `$BOOK_SPINES_DATA/derived/book-catalog/full.sqlite` (~35.6M rows) |

### Methodology

The script isolates **scoring**: every algorithm under test receives the same
FTS shortlist (permissive OR-of-OCR-tokens, `bm25` rank, cap 200). It then
argmaxes each scorer over that shortlist.

The **production** row is different — it uses `matchedTitle` /
`topCandidates[0]` from the real app telemetry run (`retrieveRoleAware` +
`FieldAwareScore` + accept policy). That is the only row reflecting the full
pipeline end-to-end.

**Hit criteria**

- *Exact*: normalized title equality
- *Loose* (exact+contains): exact **or** normalized substring containment with
  min length 4 (catches truncated titles like oracle `"Spain"` vs pick
  `"lonely planet Spain Pull-out map"`)

Reproduce:

```bash
cd book-train

# ios_en matched run
.venv/bin/python3 tools/experiment_match_algorithms.py \
  --run-json "$HOME/Library/Application Support/BookID/telemetry/runs/CFBA76E6-53F8-4F07-A8CF-051326ABA660/run.json" \
  --oracle ../optimize-gemini/fixtures/oracles/bookcase.json \
  --db "$HOME/ml/book-spines/derived/book-catalog/ios_en.sqlite"

# full matched run (~13 min eval time on full.sqlite)
.venv/bin/python3 tools/experiment_match_algorithms.py \
  --run-json "$HOME/Library/Application Support/BookID/telemetry/runs/A5C553DE-4345-40D2-A085-CD2662FACFA1/run.json" \
  --oracle ../optimize-gemini/fixtures/oracles/bookcase.json \
  --db "$HOME/ml/book-spines/derived/book-catalog/full.sqlite"
```

## Results summary

Eval wall time: **3.2 s** (ios_en DB) vs **780 s** (full DB).

| Algorithm | ios_en loose | full loose | ios_en exact | full exact | ios_en retrieval miss | full retrieval miss |
|---|---|---|---|---|---|---|
| **production (FieldAwareScore pipeline)** | **48/89 (53.9%)** | **63/89 (70.8%)** | 4/89 (4.5%) | 34/89 (38.2%) | 1/89 | 1/89 |
| `token_set_ratio` | 19/89 (21.3%) | 61/89 (68.5%) | 4/89 (4.5%) | 37/89 (41.6%) | 1/89 | 1/89 |
| `token_sort_ratio` | 13/89 (14.6%) | 57/89 (64.0%) | 3/89 (3.4%) | 37/89 (41.6%) | 1/89 | 1/89 |
| FTS rank only | 21/89 (23.6%) | 57/89 (64.0%) | 3/89 (3.4%) | 35/89 (39.3%) | 1/89 | 1/89 |
| `partial_ratio` | 15/89 (16.9%) | 60/89 (67.4%) | 0/89 | 32/89 (36.0%) | 1/89 | 1/89 |
| `WRatio` | 13/89 (14.6%) | 54/89 (60.7%) | 1/89 (1.1%) | 32/89 (36.0%) | 1/89 | 1/89 |
| **`ratio` (Levenshtein)** | **10/89 (11.2%)** | **51/89 (57.3%)** | 0/89 | 31/89 (34.8%) | 1/89 | 1/89 |
| **`jaro_winkler`** | **7/89 (7.9%)** | **45/89 (50.6%)** | 1/89 (1.1%) | 27/89 (30.3%) | 1/89 | 1/89 |

## ios_en — sample disagreements

Production vs single-blob scorers (first 15 oracle misses where picks differ):

| Spine | OCR (truncated) | Oracle title | production | token_set_ratio | WRatio | jaro_winkler |
|---|---|---|---|---|---|---|
| 2A3604F7 | lonely planet Spain Pull-out map | Spain | Spain | A lonely flute | Lonely Planet Best of Paris 2021 | Lonely Men |
| FDE7BF36 | lonely planet Argentina INCLUDES URUGUAY | Argentina | Red Planet | Planet of Exile | Lonely Planet Best of Paris 2021 | Lonely Planets |
| 7C408B28 | lonely planet Ireland Pull-out map | Ireland | Ireland | Laws, etc | Lonely Planet Best of Paris 2021 | Lonely Planets |
| 33F2C475 | 100 Best cross-country SKI TRAILS WASHIN | 100 Best Cross-Country Ski Trails in Washington | Cross-country | George Washington | Spring Is Here | 100 Best Crossword Puzzles for Adults |
| 2BAADA9C | oncly pi France Paris pull-but map | France | Paris, France | Paris, France | Paris and the Parisians in 1835 | Once in Paris |
| 6A91C497 | MAPS TO USE EASY- THE NATIONAL PARKS THE | The Complete Guide to The National Parks of the West | The national parks | Guide to national parks | Guide to national parks | The Double Eagle Guide to Camping in Western Parks and Forests |
| 248CDC49 | 8 EDITION MOUNTAINEERING The Freedom of | Mountaineering: The Freedom of the Hills | The Hills of the Dead | Age of Mountaineering the | Canada | The freedom of faith |
| C8AD1DAB | DAY HIKING SNOQUALMIE REGION | Day Hiking Snoqualmie Region | Go Hiking! | City region and regionalism | Hiking Wyoming's Wind River Range | Day |
| DA787240 | Rick Steves. PORTUGAL EDITION … FOLDO | Portugal | Portugal | Rick Steves Tour | Rick Steves' Florence and Tuscany 2008 | Rick Steves Tour |
| 3F5CE5DF | MOON VICTORIA & VLANDUVER EXPERT ADVICE | Victoria & Vancouver Island | Victoria | New Moon | Pet Expert | Moonheart |
| FA4B34C4 | BACKPACKING WASHINGTON 2nd edition | Backpacking Washington | Washington | George Washington | Dams and Appurtenant Hydraulic Structures, 2nd Edition | Backpacking |
| 12FA66DA | map pull-out Seattle Washington, Oregon | Washington, Oregon & the Pacific Northwest | The Pacific Northwest | George Washington | The Pacific Northwest | Miss Lonelyhearts & The Day of the Locust |
| 89C20424 | PACIFIC NORTHWEST MOUNTAINS NATURAL HIST | Natural History of the Pacific Northwest Mountains | The mountains | History of the Devil | The Pacific Northwest | A pictorial history of the American theatre |
| 7612768F | lonely plonet THE PACIFIC NORTHWEST'S BE | The Pacific Northwest's Best Trips | Road | The best of Saki | Lonely Road | The rise of cotton mills in the south |
| F7BE1579 | Bentley * HIKING WASHINGTON'S HISTORY | Hiking Washington's History | History | Washingtons | Go Hiking! | Meaning in history |

## full — sample disagreements

| Spine | OCR (truncated) | Oracle title | production | token_set_ratio | WRatio | jaro_winkler |
|---|---|---|---|---|---|---|
| 6FC90FB4 | 100 Best cross-country SKI TRAILS WASHIN | 100 Best Cross-Country Ski Trails in Washington | Washington | 100 best cross-country ski trails in Washington | Roads to Trails Northwest Washington | 100 best cross-country ski trails in Washington |
| 42F03DF3 | oncly pi France Paris pull-but map | France | France Map | Paris | France, Paris [Map Pack Bundle] | Lonely Planet Paris City Map |
| 86B5671C | MOON VICTORIA & VLANDUVER EXPERT ADVICE | Victoria & Vancouver Island | Expert Advice | Moon Vancouver and Victoria | Moonlight | Moon Victoria and Vancouver Island |
| 46E50B00 | BACKPACKING WASHINGTON 2nd edition | Backpacking Washington | Backpacking Washington | Backpacking | Introduction to Backpacking…, 2nd Edition | Backpacking Washington |
| A1371399 | map pull-out Seattle Washington, Oregon | Washington, Oregon & the Pacific Northwest | Lonely Planet Washington, Oregon & the Pacific Northwest | Lonely Planet Seattle | Lonely Planet Washington, Oregon and the Pacific Northwest | Lonely Planet Washington, Oregon and the Pacific Northwest |
| 1412D6A2 | lonely plonet THE PACIFIC NORTHWEST'S BE | The Pacific Northwest's Best Trips | The Pacific Northwest's best trips | The Pacific Northwest's best trips | Lonely Planet Best Road Trips USA 5 | Lonely Planet Pacific Coast Highways Road Trips 3 |
| 6BDAB522 | Bentley * HIKING WASHINGTON'S HISTORY | Hiking Washington's History | Hiking Washington's History | Hiking Washington's history | Hiking Washington's history | Walking Washington's History |
| BB4CA8EB | loneyplonet EPIC HIKES of the WORLD | Epic Hikes of the World | Epic Hikes of the World | Epic Hikes of the World | Epic Hikes of the World | Lonely Planet Epic Runs of the World |
| 8F5B2A8B | lonely planet Denmark & the Faroe Island | Denmark & the Faroe Islands | Lonely Planet Denmark and the Faroe Islands | Lonely Planet Denmark and the Faroe Islands | Lonely Planet Denmark and the Faroe Islands | Lonely Planet Denmark |
| 9F3F9821 | AFALCON GUIDE® Hiking River Gorge the Co | Hiking the Columbia River Gorge | Columbia | Hiking the Columbia River Gorge: A Guide… | Hiking the Columbia River Gorge, 2nd | Day Hikes in the Columbia River Gorge |

## Conclusions

1. **Levenshtein / Jaro-Winkler are not an upgrade.** On `ios_en` they are the
   worst scorers (8–11% loose vs 21% for `token_set_ratio`). On `full` they
   trail `token_set_ratio` and production. The shipped rerank already uses
   `tokenSetRatio` (RapidFuzz-equivalent, token-aware LCS family).

2. **Retrieval dominates rerank choice.** On `ios_en`, the experiment's simplified
   OR-token FTS shortlist caps pure scorers at ~21% loose, while the production
   pipeline (role-aware 3-pass retrieval + field split) reaches **54%**. Gains
   come from structure and shortlist quality, not swapping distance metrics.

3. **`full` inflates loose-hit rate but not usability.** Production loose hit
   rises 54% → 71%, but catalog match time rises **~5 s → ~199 s** (~40×).
   Many "hits" are substring overlaps (`"Spain"`, `"Washington"`, `"History"`)
   that feel wrong in the UI.

4. **Embedding similarity (CLIP/BERT) is unlikely to help first.** Failures are
   dominated by shared generic tokens pulling wrong books into the shortlist.
   Semantic embeddings do not fix shortlist misses and add on-device cost.

5. **Higher-leverage next steps:** match on `ios_en_shelf` (not `full` for ID),
   FM escalation for `rolesAmbiguous` spines, pass real `catalogSize` into the
   popularity term, keep strict auto-accept (90/8) and manual confirm for the rest.

## Related code

| Piece | Location |
|---|---|
| Experiment script | `tools/experiment_match_algorithms.py` |
| Production rerank | `Sources/SpineMatching/FieldAwareScore.swift` |
| Fuzzy scorers | `Sources/SpineMatching/FuzzyMatch.swift` |
| Role-aware retrieval | `Sources/SpineCatalog/BookCatalogRoleRetrieval.swift` |
| Pipeline spec | `docs/BOOK_ID_IOS_PIPELINE.md` §Catalog matching |
