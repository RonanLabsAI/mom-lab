# Run logs

Scripts write run logs here. `analysis/` holds the analysis text outputs the notebook cites. The original operational logs (queue events, generation logs, Docker
inspect dumps, host summaries) are not part of the public export. Their scrubbed SHA-256 values are
kept in `release/SCRUB-MANIFEST.json` (entries marked `excluded_from_public_export`).
