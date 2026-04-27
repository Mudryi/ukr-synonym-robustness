# Synonym data files (user-provided)

The TextFooler attack expects three external files. They are not bundled with
this repo. Place them anywhere and pass their paths via CLI flags or env vars.

| File | CLI flag | Env var | Purpose |
|---|---|---|---|
| `synonimy_info_clean.json` | `--synonym-dict` | `UKR_SYN_DICT` | Required. Main synonym dictionary. |
| `hand_parsed_top_100.json` | `--hand-parsed` | `UKR_SYN_HAND` | Optional. Hand-curated additions, merged on load. |
| `antonimy.jsonlines` | `--antonyms` | `UKR_SYN_ANTONYMS` | Optional. Adds extra synonyms from antonym entries; removes antonyms from synonym lists. |

## Expected schemas

### `synonimy_info_clean.json` (JSON array)
```json
[
  {
    "lemma": "великий",
    "synsets": [
      {"clean": ["величезний", "грандіозний"]}
    ]
  },
  ...
]
```

### `hand_parsed_top_100.json` (JSON dict)
```json
{
  "великий": ["величезний", "грандіозний"],
  ...
}
```

### `antonimy.jsonlines` (one JSON entry per line)
```json
{"lemma": "великий", "synonyms": ["величезний"], "antonyms": ["маленький"], "url": "...", "samples": []}
```
