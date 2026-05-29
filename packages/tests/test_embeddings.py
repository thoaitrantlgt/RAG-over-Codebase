import importlib.util
import os
import unittest

from packages.indexing.embedding import (
    FastEmbedEmbeddingProvider,
    HashingEmbeddingProvider,
    create_embedding_provider,
)


class EmbeddingProviderTests(unittest.TestCase):
    def test_create_hashing_embedding_provider(self) -> None:
        provider = create_embedding_provider(provider="hashing", dimensions=32)
        vector = provider.embed("verify token")

        self.assertIsInstance(provider, HashingEmbeddingProvider)
        self.assertEqual(provider.dimensions, 32)
        self.assertEqual(len(vector), 32)

    @unittest.skipUnless(
        importlib.util.find_spec("fastembed") and os.getenv("RUN_FASTEMBED_TESTS"),
        "Set RUN_FASTEMBED_TESTS=1 to instantiate and download a real FastEmbed model.",
    )
    def test_fastembed_provider_reports_real_model_dimensions(self) -> None:
        provider = FastEmbedEmbeddingProvider(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="data/models/fastembed-test",
        )

        self.assertEqual(provider.dimensions, 384)


if __name__ == "__main__":
    unittest.main()
