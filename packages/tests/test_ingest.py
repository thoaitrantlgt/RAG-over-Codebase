import json
import tempfile
import unittest
from pathlib import Path

from packages.ingestion import ingest_repository, write_jsonl


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


class IngestionTests(unittest.TestCase):
    def test_extracts_python_and_typescript_symbols(self) -> None:
        result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")

        self.assertEqual(result.errors, [])
        self.assertEqual(result.files_scanned, 3)

        names = [chunk.symbol_name for chunk in result.chunks]
        self.assertEqual(
            names,
            [
                "check_permission",
                "AuthService",
                "AuthService.verify_token",
                "AuthService.refresh_token",
                "UserCard",
                "getUser",
                "UserService",
                "UserService.findUser",
                "ExportedService",
                "ExportedService.run",
            ],
        )

    def test_python_methods_include_qualified_class_names_and_exact_code(self) -> None:
        result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")
        method = next(
            chunk
            for chunk in result.chunks
            if chunk.symbol_name == "AuthService.verify_token"
        )

        self.assertEqual(method.path, "src/auth/service.py")
        self.assertEqual(method.language, "python")
        self.assertEqual(method.symbol_kind, "method")
        self.assertEqual(method.start_line, 9)
        self.assertEqual(method.end_line, 11)
        self.assertIn("def verify_token", method.code_body)
        self.assertEqual(
            method.chunk_id,
            "sample_repo:src/auth/service.py:AuthService.verify_token:9:11",
        )
        self.assertEqual(len(method.content_hash), 64)

    def test_typescript_classes_and_methods_are_extracted(self) -> None:
        result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")
        by_name = {chunk.symbol_name: chunk for chunk in result.chunks}

        self.assertEqual(by_name["ExportedService"].symbol_kind, "class")
        self.assertEqual(by_name["ExportedService.run"].symbol_kind, "method")
        self.assertEqual(by_name["ExportedService"].language, "typescript")

    def test_write_jsonl_writes_one_record_per_line(self) -> None:
        result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "chunks.jsonl"
            write_jsonl(out, result.chunks[:2])
            lines = out.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["repo"], "sample_repo")


if __name__ == "__main__":
    unittest.main()
