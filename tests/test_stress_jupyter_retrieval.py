from __future__ import annotations

import unittest

from scripts.stress_jupyter_retrieval import (
    RESULT_MARKER,
    VirtualUserResult,
    _extract_result,
    find_setup_code,
    parse_waves,
    summarize_wave,
)


class JupyterStressHelpersTests(unittest.TestCase):
    def test_parse_waves(self) -> None:
        self.assertEqual(parse_waves("5, 10,20,40"), [5, 10, 20, 40])
        with self.assertRaises(ValueError):
            parse_waves("5,0")

    def test_extract_result_selects_requested_phase(self) -> None:
        stdout = "noise\n" + RESULT_MARKER + '{"phase":"search","hits":8,"search_s":0.1}\n'
        self.assertEqual(_extract_result(stdout, phase="search")["hits"], 8)

    def test_torch_thread_limits_precede_notebook_setup(self) -> None:
        notebook = {
            "cells": [{
                "cell_type": "code",
                "source": "retriever = RetrievalClient.from_env(cache_dir='cache')\n",
            }]
        }
        _, code = find_setup_code(
            notebook,
            setup_cell=None,
            include_install=False,
            torch_threads=2,
            torch_interop_threads=2,
        )
        self.assertLess(code.index("torch.set_num_threads(2)"), code.index("retriever ="))
        self.assertLess(
            code.index("torch.set_num_interop_threads(2)"),
            code.index("retriever ="),
        )

    def test_summarize_wave(self) -> None:
        results = [
            VirtualUserResult(
                wave=2,
                user_id=1,
                ok=True,
                kernel_create_s=0.2,
                setup_s=2.0,
                model_load_s=1.5,
                search_latencies_s=[0.1, 0.2],
                rss_mb=100.0,
            ),
            VirtualUserResult(
                wave=2,
                user_id=2,
                ok=False,
                kernel_create_s=0.3,
                error_phase="setup",
                error="out of memory",
            ),
        ]

        summary = summarize_wave(results, before_status=None, after_status=None)

        self.assertEqual(summary["users"], 2)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["search_s"]["count"], 2)
        self.assertEqual(summary["error_phases"], {"setup": 1})


if __name__ == "__main__":
    unittest.main()
