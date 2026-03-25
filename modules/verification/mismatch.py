"""
Cargo Mismatch Detector - CargoGuard SENTINEL (Module 2)
Verify cargo contents match declared manifest type.
"""

import json
import numpy as np
from pathlib import Path

try:
    from modules.verification.build_embeddings import EmbeddingExtractor
except ImportError:
    from build_embeddings import EmbeddingExtractor


class CargoVerifier:
    THRESHOLD = 0.55

    def __init__(self, embeddings_path: str, device: str = 'cpu'):
        self.extractor = EmbeddingExtractor(device=device)
        with open(embeddings_path) as f:
            raw = json.load(f)
        self.references = {k: np.array(v) for k, v in raw.items()}
        print(f'CargoVerifier loaded. Categories: {list(self.references.keys())}')

    def verify(self, image_path: str, declared_type: str) -> dict:
        embedding = self.extractor.extract(image_path)
        similarities = {}
        for cat, ref_vec in self.references.items():
            sim = float(np.dot(embedding, ref_vec))
            similarities[cat] = round(sim, 4)

        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]
        declared_score = similarities.get(declared_type, 0.0)

        is_mismatch = (
            declared_score < self.THRESHOLD or
            (best_match != declared_type and best_score - declared_score > 0.15)
        )
        mismatch_score = int((1 - declared_score) * 100) if is_mismatch else 0

        explanation = ''
        if is_mismatch:
            explanation = f"Mismatch: declared '{declared_type}' but looks like '{best_match}'"

        return {
            'declared_type': declared_type,
            'best_match': best_match,
            'declared_score': round(declared_score, 4),
            'best_score': round(best_score, 4),
            'all_similarities': similarities,
            'is_mismatch': is_mismatch,
            'mismatch_score': mismatch_score,
            'explanation': explanation,
        }
