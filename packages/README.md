# AST-Aware Ingestion: Python

Phase 1 scans a source repository, parses supported files with Tree-sitter, extracts syntax-aware code chunks, and writes JSONL records.

## Supported Languages

- Python: `.py`
- TypeScript: `.ts`
- TSX: `.tsx`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m packages.ingestion.cli --repo-path ./sample-repo --repo-name sample-repo --out data/chunks/sample.jsonl
```

## Test

```bash
python -m unittest discover -s packages/tests
```

## Output Contract

```json
{
  "repo": "my-backend-repo",
  "path": "src/auth/service.py",
  "language": "python",
  "start_line": 42,
  "end_line": 68,
  "symbol_name": "AuthService.verify_token",
  "symbol_kind": "method",
  "code_body": "def verify_token(token):\n    ...",
  "chunk_id": "my-backend-repo:src/auth/service.py:AuthService.verify_token:42:68",
  "content_hash": "..."
}
```
