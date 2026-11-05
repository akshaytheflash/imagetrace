"""Unit tests (stdlib only, no pytest dependency).

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edittrace.config import Config
from edittrace.explain.report import render_report
from edittrace.forensics.stack import build_features
from edittrace.io_utils import bundle_from_array, load_image, thumbnail_similarity
from edittrace.models.unet import UNet, count_params
from edittrace.ordering.cues import PairEvidence
from edittrace.ordering.topo import kendall_tau, order_from_evidence
from edittrace.pipeline import analyze
from edittrace.provenance.audit import audit
from edittrace.regions.extract import extract_regions, region_mask_from_prob
from edittrace.synth import make_tampered, write_jpeg_with_exif


class TestSynthesis(unittest.TestCase):
    def test_scene_and_edits(self):
        s = make_tampered(3, size=(160, 200), n_edits=2)
        self.assertEqual(s.bgr.shape, (160, 200, 3))
        self.assertEqual(s.bgr.dtype, np.uint8)
        self.assertEqual(s.mask.shape, (160, 200))
        self.assertGreater(int(s.mask.sum()), 0)
        self.assertEqual([e.order for e in s.edits], list(range(len(s.edits))))
        for e in s.edits:
            self.assertIsNotNone(e.mask)
            self.assertGreater(e.area_frac, 0)

    def test_clean_sample(self):
        s = make_tampered(4, size=(128, 128), n_edits=0)
        self.assertEqual(len(s.edits), 0)
        self.assertEqual(int(s.mask.sum()), 0)


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = make_tampered(5, size=(128, 160), n_edits=1)
        cls.bundle = bundle_from_array(s.bgr)
        cls.stack = build_features(cls.bundle)

    def test_tensor_shape_and_range(self):
        t = self.stack.tensor()
        self.assertEqual(t.shape, (18, 128, 160))
        self.assertTrue(np.isfinite(t).all())
        self.assertGreaterEqual(float(t.min()), -0.01)
        self.assertLessEqual(float(t.max()), 1.01)

    def test_maps_present(self):
        for name in ("ela", "noise_z", "ghost_min", "seam", "cfa_resid"):
            self.assertIn(name, self.stack.maps)
            self.assertEqual(self.stack.maps[name].shape, (128, 160))

    def test_ela_is_zero_for_identical_recompression(self):
        # A synthetic flat image recompressed at its own quality has very low
        # residual compared with a hard content edge.
        flat = np.full((64, 64, 3), 128, np.uint8)
        from edittrace.forensics.ela import ela_map
        raw, _ = ela_map(flat, quality=95)
        self.assertLess(float(raw.mean()), 1.0)


class TestRegions(unittest.TestCase):
    def test_threshold_extracts_blob(self):
        prob = np.zeros((120, 160), np.float32)
        prob[30:80, 50:110] = 0.95
        cfg = Config()
        binary = region_mask_from_prob(prob, cfg)
        self.assertGreater(int(binary.sum()), 0)
        regions, _ = extract_regions(build_features(bundle_from_array(
            make_tampered(6, size=(120, 160), n_edits=0).bgr)), prob, cfg)
        self.assertEqual(len(regions), 1)
        x, y, w, h = regions[0].bbox
        self.assertLess(abs(x - 50), 14)
        self.assertLess(abs(y - 30), 14)

    def test_empty_prob_gives_no_regions(self):
        prob = np.zeros((96, 96), np.float32)
        regions, binary = extract_regions(
            build_features(bundle_from_array(make_tampered(7, size=(96, 96), n_edits=0).bgr)),
            prob, Config())
        self.assertEqual(regions, [])
        self.assertEqual(int(binary.sum()), 0)


class TestOrdering(unittest.TestCase):
    def test_respects_pairwise_scores(self):
        pairs = [
            PairEvidence(a=1, b=2, score=0.8, cues={"compression_lineage": 0.8}),
            PairEvidence(a=2, b=3, score=0.6, cues={"seam_truncation": 0.6}),
            PairEvidence(a=1, b=3, score=0.4, cues={"noise_floor": 0.4}),
        ]
        res = order_from_evidence(pairs, Config())
        self.assertEqual(res.sequence, [1, 2, 3])
        self.assertGreater(res.confidence, 0.2)

    def test_cycle_is_broken_not_crashed(self):
        pairs = [
            PairEvidence(a=1, b=2, score=0.9, cues={}),
            PairEvidence(a=2, b=3, score=0.9, cues={}),
            PairEvidence(a=3, b=1, score=0.9, cues={}),
        ]
        res = order_from_evidence(pairs, Config())
        self.assertEqual(sorted(res.sequence), [1, 2, 3])
        self.assertGreaterEqual(res.cycles_broken, 1)

    def test_undecided_pairs_reported(self):
        pairs = [PairEvidence(a=1, b=2, score=0.01, cues={})]
        res = order_from_evidence(pairs, Config())
        self.assertEqual(len(res.undecided), 1)
        self.assertEqual(res.confidence, 0.0)

    def test_kendall_tau(self):
        self.assertAlmostEqual(kendall_tau([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(kendall_tau([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertTrue(np.isnan(kendall_tau([1], [1])))


class TestProvenance(unittest.TestCase):
    def test_editor_software_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "edited.jpg"
            bgr = make_tampered(8, size=(96, 128), n_edits=1).bgr
            write_jpeg_with_exif(bgr, p, software="Adobe Photoshop 25.0")
            b = load_image(p)
            a = audit(b)
            ids = [f.id for f in a.findings]
            self.assertIn("software_tag", ids)
            self.assertLess(a.score, 0.8)

    def test_plain_export_flags_missing_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "plain.jpg"
            cv2.imwrite(str(p), make_tampered(9, size=(96, 128), n_edits=1).bgr)
            a = audit(load_image(p))
            self.assertIn("metadata_stripped", [f.id for f in a.findings])

    def test_thumbnail_similarity(self):
        img = make_tampered(10, size=(128, 128), n_edits=0).bgr
        small = cv2.resize(img, (64, 64))
        self.assertGreater(thumbnail_similarity(img, small), 0.9)
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.assertLess(abs(thumbnail_similarity(img, noise)), 0.5)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = make_tampered(11, size=(144, 176), n_edits=2)
        cls.trace = analyze(cls.sample.bgr, Config(), use_model=False)

    def test_trace_shape(self):
        self.assertEqual(self.trace.prob.shape, (144, 176))
        self.assertEqual(self.trace.mask.shape, (144, 176))
        self.assertEqual(self.trace.bundle.w, 176)

    def test_trace_is_json_serialisable(self):
        d = self.trace.to_dict()
        text = json.dumps(d, default=float)
        self.assertIn("regions", json.loads(text))
        self.assertEqual(d["image"]["width"], 176)

    def test_report_renders(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.trace.render_html(Path(td) / "r.html")
            html = out.read_text(encoding="utf-8")
            self.assertIn("Edit order", html)
            self.assertIn("Provenance audit", html)
            self.assertIn("trace-data", html)
            self.assertGreater(len(html), 5000)

    def test_model_forward_shape(self):
        import torch
        m = UNet(in_ch=18, width=8)
        self.assertGreater(count_params(m), 10_000)
        x = torch.zeros(1, 18, 64, 64)
        self.assertEqual(tuple(m(x).shape), (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
