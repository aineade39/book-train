# Google Books API vs spine OCR (bookcase.jpg)

> **Live API note:** Without ``GOOGLE_BOOKS_API_KEY``, the shared anonymous
> quota returns HTTP 429. Use ``--fixtures tools/fixtures/google_books_responses.json``
> for offline replay, or set a Books API key and re-run without ``--fixtures``.

Run: `/Users/joebr/Library/Application Support/BookID/telemetry/runs/CFBA76E6-53F8-4F07-A8CF-051326ABA660/run.json`  
Oracle: `/Users/joebr/dev/optimize-gemini/fixtures/oracles/bookcase.json`  
Spines evaluated: **25**
Mode: **live API**

## Summary

| Metric | Count |
|---|---|
| API returned a title | 25/25 |
| API title ↔ oracle fuzzy ≥ 70 | 24/25 |
| OCR ↔ oracle fuzzy ≥ 70 | 24/25 |
| API clearly better than OCR (+5 fuzzy vs oracle) | 2/25 |
| OCR clearly better than API (+5 fuzzy vs oracle) | 1/25 |

## Query strategy

Tiered queries per spine (first hit pool merged, best ``token_set_ratio`` wins):

1. ``intitle_phrase`` — quoted longest significant token run from OCR
2. ``intitle_tokens`` — up to 3 ``intitle:`` clauses on longest tokens
3. ``fulltext`` — normalized OCR as general ``q=`` (truncated)

API params: ``printType=books``, ``langRestrict=en``, ``orderBy=relevance``, ``maxResults=5``, ``projection=full``.

## Full results

| Spine | OCR | Oracle title | Google Books title | Author | mainCategory | categories | rating (count) | Strategy | OCR↔API | API↔Oracle | OCR↔Oracle |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| 2A3604F7 | lonely planet Spain Pull-out map | Spain | Lonely Planet Spain | Lonely Planet, Gregor Clark, Duncan Garwood, Anthony Ham, Catherine Le Nevez, John Noble, Josephine Quintero, Brendan Sainsbury, Regis St Louis, Andy Symington, Sally Davies, Isabella Noble | — | Travel | — | fulltext | 75 | 100 | 100 |
| FDE7BF36 | lonely planet Argentina INCLUDES URUGUAY | Argentina | Lonely Planet Argentina (Travel Guide) | Lonely Planet | — | Travel | — | fulltext | 79 | 100 | 100 |
| 7C408B28 | lonely planet Ireland Pull-out map | Ireland | Lonely Planet Ireland Planning Map | Planet Lonely | — | Reference | — | fulltext | 85 | 100 | 100 |
| 33F2C475 | 100 Best cross-country SKI TRAILS WASHINGTON Third Edition KIRKENDALL SPRING | 100 Best Cross-Country Ski Trails in Washington | 100 Best Cross-Country Ski Trails in Washington | Tom Kirkendall, Vicky Spring | — | Sports & Recreation | — | fulltext | 91 | 100 | 97 |
| 2BAADA9C | oncly pi France Paris pull-but map | France | The Illustrated London News | — | — | Great Britain | — | fulltext | 33 | 24 | 100 |
| 6A91C497 | MAPS TO USE EASY- THE NATIONAL PARKS THE COMPLETE GUIDE TO OF THE WEST Fodors | The Complete Guide to The National Parks of the West | Fodor's The Complete Guide to the National Parks of the West | Fodor's Travel Guides | — | Travel | — | fulltext | 83 | 100 | 100 |
| 248CDC49 | 8 EDITION MOUNTAINEERING The Freedom of the Hills | Mountaineering: The Freedom of the Hills | Mountaineering | — | — | Sports & Recreation | — | intitle_tokens | 100 | 100 | 100 |
| C8AD1DAB | DAY HIKING SNOQUALMIE REGION | Day Hiking Snoqualmie Region | Day Hiking Snoqualmie Region | Dan Nelson | — | Sports & Recreation | — | intitle_phrase | 100 | 100 | 100 |
| DA787240 | Rick Steves. PORTUGAL EDITION П1тH FOLDOUT | Portugal | Rick Steves Portugal | Rick Steves | — | Travel | — | fulltext | 79 | 100 | 100 |
| F4A2CBD2 | HISTORY OF A TRAVELLER'S Ireland | Ireland | A Traveller's History of Ireland | Peter Neville | — | History | — | intitle_tokens | 100 | 100 | 100 |
| 3F5CE5DF | MOON VICTORIA & VLANDUVER EXPERT ADVICE | Victoria & Vancouver Island | Moon Victoria & Vancouver Island | Andrew Hempstead | — | Travel | — | fulltext | 73 | 100 | 64 |
| FA4B34C4 | BACKPACKING WASHINGTON 2nd edition | Backpacking Washington | Backpacking Washington | Douglas Lorain | — | Travel | — | intitle_phrase | 79 | 100 | 100 |
| 12FA66DA | map pull-out Seattle Washington, Oregon & the Pacific Northwest lonely planet | Washington, Oregon & the Pacific Northwest | Washington, Oregon & Pacific Northwest | Lonely Planet | — | Travel | — | fulltext | 100 | 100 | 100 |
| 89C20424 | PACIFIC NORTHWEST MOUNTAINS NATURAL HISTORY of the DANIEL MATHEWS | Natural History of the Pacific Northwest Mountains | Natural History of the Pacific Northwest Mountains | Daniel Mathews | — | Nature | — | intitle_tokens | 100 | 100 | 100 |
| 7612768F | lonely plonet THE PACIFIC NORTHWEST'S BEST TRIPS 32 AMAZINO ROAD TRIPS + PULL OUT MAP | The Pacific Northwest's Best Trips | Lonely Planet Road Trips the Pacific Northwest's Best Trips | Ryan Ver Berkmoes | — | Automobile travel | — | fulltext | 79 | 100 | 100 |
| F7BE1579 | Bentley * HIKING WASHINGTON'S HISTORY | Hiking Washington's History | Hiking Washington's History | Judy Bentley | — | Sports & Recreation | — | intitle_phrase | 100 | 100 | 100 |
| 8BB4288A | loneyplonet EPIC HIKES of the WORLD | Epic Hikes of the World | What Are the Epic Hikes of the World | Peter Andy | — | Travel | — | fulltext | 79 | 100 | 100 |
| 6EC784E2 | DAY HIKING CENTRAL CASCADES | Day Hiking Central Cascades | Day Hiking Central Cascades | Craig Romano | — | Sports & Recreation | — | intitle_phrase | 100 | 100 | 100 |
| E057E353 | lonely planet Denmark & the Faroe Islands | Denmark & the Faroe Islands | Denmark & the Faroe Islands | Lonely Planet | — | — | — | intitle_tokens | 100 | 100 | 100 |
| C4446AFB | AFALCON GUIDE® Hiking River Gorge the Columbia | Hiking the Columbia River Gorge | Hiking the Columbia River Gorge, 3rd | Jim Yuskavitch | — | Sports & Recreation | — | fulltext | 81 | 100 | 100 |
| 4399DE70 | DAY HIKING EASTERN WASHINGTON | Day Hiking Eastern Washington | Day Hiking Eastern Washington | Rich Landers, Craig Romano | — | Sports & Recreation | — | intitle_phrase | 100 | 100 | 100 |
| 20223AC5 | IRELAND'S BEST WALKS A WALKING GUIDE HELEN FAIRBAIRN | Ireland's Best Walks | Ireland's Best Walks | Helen Fairbairn | — | Sports & Recreation | — | fulltext | 100 | 100 | 100 |
| 327900A1 | World Food MEXICO | World Food Mexico | World Food: Mexico City | James Oseland | — | Cooking | — | intitle_phrase | 100 | 100 | 100 |
| 92E00E0C | ROMANO URBAN TRAILS EASTSIDE | Urban Trails Eastside | Urban Trails: Eastside | Craig Romano | — | Sports & Recreation | — | fulltext | 100 | 100 | 100 |
| 33D493D9 | WASHINGTON'S GLACIER PEAK REGION 100*· | 100 Hikes Washington's Glacier Peak Region | 100 Hikes in Washington's Glacier Peak Region | Ira Spring, Harvey Manning | — | Cascade Range | — | intitle_tokens | 93 | 100 | 93 |

## Notable rows

### API win (API↔Oracle ≥ 70 and beats OCR by ≥ 5)

- **3F5CE5DF** OCR: `MOON VICTORIA & VLANDUVER EXPERT ADVICE`
  - Oracle: *Victoria & Vancouver Island*
  - API: *Moon Victoria & Vancouver Island* — Andrew Hempstead
  - Fuzzy: OCR↔API 73, API↔Oracle 100, OCR↔Oracle 64
- **33D493D9** OCR: `WASHINGTON'S GLACIER PEAK REGION 100*·`
  - Oracle: *100 Hikes Washington's Glacier Peak Region*
  - API: *100 Hikes in Washington's Glacier Peak Region* — Ira Spring, Harvey Manning
  - Fuzzy: OCR↔API 93, API↔Oracle 100, OCR↔Oracle 93

### API miss (no result or API↔Oracle < 40)

- **2BAADA9C** OCR: `oncly pi France Paris pull-but map`
  - Oracle: *France*
  - API: *The Illustrated London News* — 
  - Fuzzy: OCR↔API 33, API↔Oracle 24, OCR↔Oracle 100

### OCR was already close (OCR↔Oracle ≥ 70)

- **2A3604F7** OCR: `lonely planet Spain Pull-out map`
  - Oracle: *Spain*
  - API: *Lonely Planet Spain* — Lonely Planet, Gregor Clark, Duncan Garwood, Anthony Ham, Catherine Le Nevez, John Noble, Josephine Quintero, Brendan Sainsbury, Regis St Louis, Andy Symington, Sally Davies, Isabella Noble
  - Fuzzy: OCR↔API 75, API↔Oracle 100, OCR↔Oracle 100
- **FDE7BF36** OCR: `lonely planet Argentina INCLUDES URUGUAY`
  - Oracle: *Argentina*
  - API: *Lonely Planet Argentina (Travel Guide)* — Lonely Planet
  - Fuzzy: OCR↔API 79, API↔Oracle 100, OCR↔Oracle 100
- **7C408B28** OCR: `lonely planet Ireland Pull-out map`
  - Oracle: *Ireland*
  - API: *Lonely Planet Ireland Planning Map* — Planet Lonely
  - Fuzzy: OCR↔API 85, API↔Oracle 100, OCR↔Oracle 100
- **33F2C475** OCR: `100 Best cross-country SKI TRAILS WASHINGTON Third Edition KIRKENDALL S…`
  - Oracle: *100 Best Cross-Country Ski Trails in Washington*
  - API: *100 Best Cross-Country Ski Trails in Washington* — Tom Kirkendall, Vicky Spring
  - Fuzzy: OCR↔API 91, API↔Oracle 100, OCR↔Oracle 97
- **2BAADA9C** OCR: `oncly pi France Paris pull-but map`
  - Oracle: *France*
  - API: *The Illustrated London News* — 
  - Fuzzy: OCR↔API 33, API↔Oracle 24, OCR↔Oracle 100
- **6A91C497** OCR: `MAPS TO USE EASY- THE NATIONAL PARKS THE COMPLETE GUIDE TO OF THE WEST …`
  - Oracle: *The Complete Guide to The National Parks of the West*
  - API: *Fodor's The Complete Guide to the National Parks of the West* — Fodor's Travel Guides
  - Fuzzy: OCR↔API 83, API↔Oracle 100, OCR↔Oracle 100
- **248CDC49** OCR: `8 EDITION MOUNTAINEERING The Freedom of the Hills`
  - Oracle: *Mountaineering: The Freedom of the Hills*
  - API: *Mountaineering* — 
  - Fuzzy: OCR↔API 100, API↔Oracle 100, OCR↔Oracle 100
- **C8AD1DAB** OCR: `DAY HIKING SNOQUALMIE REGION`
  - Oracle: *Day Hiking Snoqualmie Region*
  - API: *Day Hiking Snoqualmie Region* — Dan Nelson
  - Fuzzy: OCR↔API 100, API↔Oracle 100, OCR↔Oracle 100

