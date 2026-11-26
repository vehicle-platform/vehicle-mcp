# Retrieval Quality Evaluation Dataset (`search_knowledge`, G4-7)

Evaluates `vehicle-mcp`'s hybrid FTS5 + `sqlite-vec` (`multilingual-e5-small`, RRF) retrieval over
the indexed fault catalog and platform idea docs (`vehicle-defs/docs/ideas/`).

## Dataset Composition

Per locale (`en` in `queries.json`, `tr` in `locales/tr.json`), paired 1:1 across locales with
identical expected targets and splits:

- **174 answerable queries per locale** (104 pre-existing + **70 newly authored** in G4-7):
  - `old_catalog` (32): authored fault-catalog queries from G2-2 (`en-01..32` / `tr-01..32`).
  - `old_docs` (10): authored doc-section queries from G2-2 (`en-33..42` / `tr-33..42`).
  - `old_holdout` (62): verbatim fault-catalog complaint phrases (`ho-en-*` / `ho-tr-*`).
  - `new_catalog` (35): newly authored rider-phrased fault questions covering all 31 displayable
    catalog faults (`en-g4-01..35` / `tr-g4-01..35`).
  - `new_docs` (35): newly authored engineering/platform doc questions covering all indexed
    `docs/ideas/*.md` files (`en-g4-36..70` / `tr-g4-36..70`).
- **14 weak (out-of-domain) queries per locale** (6 pre-existing + 8 newly authored) for noise
  checks.

All newly authored queries were written directly from the fault catalog and `docs/ideas/*.md`
sources without reusing any text from `vehicle-ml`'s held-out eval or `vehicle-server`'s
`chat-eval` suites.

## Fixed-Seed Split Procedure (`seed = 20261127`)

All items (old + new) are split **50 / 50** into `tune` and `test` stratified by stratum
(`new_catalog`, `new_docs`, `old_catalog`, `old_docs`, `old_holdout`, and `weak_queries`) using
Python's `random.Random(20261127)` shuffling within each sorted stratum (taking `len(rows) // 2`
for `tune` and the remainder for `test`). Both locales share the exact same split assignment per
paired item.

| Stratum | `tune` | `test` | Total |
|---|---:|---:|---:|
| `new_catalog` | 17 | 18 | 35 |
| `new_docs` | 17 | 18 | 35 |
| `old_catalog` | 16 | 16 | 32 |
| `old_docs` | 5 | 5 | 10 |
| `old_holdout` (catalog complaint phrases) | 31 | 31 | 62 |
| **Answerable total (per locale)** | **86** | **88** | **174** |
| `weak_queries` (out-of-domain) | 7 | 7 | 14 |

## Evaluation Protocol (Read-Only `test` Rule)

- **Never tune retrieval weights, cutoffs, synonym tables, or chunking on `test`.** Use `tune`
  for any development iteration, and report final metrics on `test` only.
- Always build the evaluation index with `--exclude-holdout` so verbatim catalog complaint phrases
  in `test` (and `holdout`) are removed from the indexed complaint chunks and fault `Complaints:`
  lines before scoring (preventing exact-match inflation).

```bash
# 1. Build a temporary evaluation index (excludes verbatim test/holdout complaint phrases)
VEHICLE_SERVER_URL=http://127.0.0.1:8000 VEHICLE_API_TOKEN=... \
  uv run --extra rag vehicle-mcp-index build \
    --db /tmp/rag-eval.sqlite --rebuild --exclude-holdout \
    --docs ../vehicle-defs/docs/ideas

# 2. Measure recall@1, recall@5 and MRR per locale on `test` only
uv run --extra rag vehicle-mcp-index eval \
  --db /tmp/rag-eval.sqlite --split test --locales en tr --show-misses
```
