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

## Split Procedure (`seed = 20261127`)

The shipped retrieval parameters (`min_similarity`, `lexical_top_n`, see `tuning.json`) were chosen
on the 42 answerable + 6 weak pre-existing `tune` items and vetoed on the 62 pre-existing `holdout` items. Those
items are therefore not clean for measurement, so **each old item keeps its original split**
(`tune` stays `tune`, `holdout` stays `holdout`) and `test` holds **only new G4-7 items**.

The 70 new answerable items and 8 new weak items per locale (ids containing `-g4-`) are split
50 / 50 into `tune` and `test`, stratified by stratum (`new_catalog`, `new_docs`, new weak), with
`random.Random(20261127)` shuffling each sorted stratum and taking `len(rows) // 2` for `tune`.
Both locales share the same assignment per paired item.

| Stratum | `tune` | `test` | `holdout` | Total |
|---|---:|---:|---:|---:|
| old answerable (authored, G2-2) | 42 | 0 | 0 | 42 |
| old `holdout` (catalog complaint phrases) | 0 | 0 | 62 | 62 |
| `new_catalog` | 17 | 18 | 0 | 35 |
| `new_docs` | 17 | 18 | 0 | 35 |
| **Answerable total (per locale)** | **76** | **36** | **62** | **174** |
| weak (out-of-domain), 6 old + 8 new | 10 | 4 | 0 | 14 |

`test` is small (36 per locale), so differences of a few points between runs are noise.

## Evaluation Protocol (Read-Only `test` Rule)

- **Never tune retrieval weights, cutoffs, synonym tables, or chunking on `test`.** Use `tune`
  for any development iteration, and report final metrics on `test` only.
- Always build the evaluation index with `--exclude-holdout` so verbatim catalog complaint phrases
  in the `holdout` and `test` splits are removed from the indexed complaint chunks and fault `Complaints:`
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

## Measured results (`test` split, current retrieval, measure only)

Index built from the fault catalog served by a local `vehicle-server` (defs v0.33.0, 31 displayable
faults plus unconfirmed `data_only` entries) and `vehicle-defs/docs/ideas`, with `--exclude-holdout`;
commands as above, 2026-11-27. n = 36 answerable queries per locale, so treat differences of a few
points as noise.

| Locale | n | recall@1 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| en | 36 | 0.583 | 0.917 | 0.734 |
| tr | 36 | 0.194 | 0.417 | 0.279 |

Observations (not acted on in this task): the `faults/unconfirmed#data only` catalog chunk often
ranks first on Turkish and short fault queries, and Turkish queries over the English-only
`docs/ideas` chunks are mostly vector-only hits. Any fix is a separate task tuned on `tune` only.

Note: `en-02` (`tune`) was reworded in G4-7 because its old text was identical to the `holdout`
phrase `ho-en-ENG-001-2`; the `tune` numbers in `tuning.json` predate this and move by at most one item.
