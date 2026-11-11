# Security

`ranking_metrics` parses text inputs supplied by the caller and, through the CLI, reads files named on the command line.

## Defended

- `parse_qrels` and `parse_run` reject duplicate `(qid, docid)` pairs with `RankingError('duplicate_entry', ...)`.
- Lines that do not contain exactly four whitespace-separated fields in qrels or six in run raise `RankingError('parse_error', ...)` with the line number.
- Relevance scores in qrels must parse as integers; scores in runs must parse as floats.
- Both parsers enforce `max_lines` (default `5_000_000`) and raise `RankingError('input_too_large', ...)` when the line count is exceeded.
- Error messages quote at most 200 characters of an offending line via `_excerpt`.
- The library makes no network or subprocess calls. The CLI performs local file reads only.
- The package logger uses `logging.getLogger(__name__)` with a `NullHandler`, so the library does not emit log records by default.

## Not defended

- There is no byte-size limit. `parse_qrels` and `parse_run` call `text.splitlines()` before checking `max_lines`, and the CLI reads the whole file with `f.read()`, so a very large input or a single very long line can still consume memory.
- The line limit does not validate file paths, file permissions, or whether files are trusted.
- There is no authentication, authorization, sandboxing, or privilege separation; callers and CLI users control their own inputs and files.
- `parse_run` accepts `float()` values such as `nan` and `inf`; the library does not reject non-finite scores or otherwise sanitize ranking scores.
- The library does not protect against misleading or malicious qrels content, only against malformed field counts, invalid score types, duplicate pairs, and excessive line counts.
