import tempfile
import unittest
from pathlib import Path

from packages.sync.incremental import run_incremental_sync


class SyncTests(unittest.TestCase):
    def test_incremental_sync_tracks_added_updated_and_deleted_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            repo.mkdir()
            service = repo / "service.py"
            service.write_text(
                "def alpha():\n    return 1\n\n"
                "def beta():\n    return 2\n",
                encoding="utf-8",
            )
            state = root / "state.json"

            first = run_incremental_sync(
                repo_path=repo,
                repo_name="repo",
                state_path=state,
                changed_chunks_out=root / "changed1.jsonl",
            )
            self.assertEqual(first.files_added, ["service.py"])
            self.assertEqual(len(first.chunks_added), 2)

            service.write_text(
                "def alpha():\n    return 10\n\n"
                "def gamma():\n    return 3\n",
                encoding="utf-8",
            )
            second = run_incremental_sync(
                repo_path=repo,
                repo_name="repo",
                state_path=state,
                changed_files=["service.py"],
                changed_chunks_out=root / "changed2.jsonl",
            )

        self.assertEqual(second.files_modified, ["service.py"])
        self.assertTrue(any("alpha" in chunk_id for chunk_id in second.chunks_updated))
        self.assertTrue(any("gamma" in chunk_id for chunk_id in second.chunks_added))
        self.assertTrue(any("beta" in chunk_id for chunk_id in second.chunks_deleted))


if __name__ == "__main__":
    unittest.main()
