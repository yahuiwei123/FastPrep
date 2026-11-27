"""Concurrent evidence generation preserves ordering and all QC decisions."""

import copy
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
import qc_runtime


class GroupParallelTests(unittest.TestCase):
    def test_parallel_checks_match_serial_and_failures_are_preserved(self):
        node = {'node_id': 'group_qc', 'policy': {'checked_nodes': ['first', 'second']}}
        barrier = threading.Barrier(2)
        parallel = False

        class FakeEvidence:
            def __init__(self, catalog, node, runtime, output, stage):
                self.stage = stage
                self.root = Path(output) / stage
                self.root.mkdir(parents=True)

            def generate(self):
                if parallel:
                    barrier.wait(timeout=5)
                status = 'fail' if self.stage == 'first' else 'unknown'
                qc_runtime.write_json(self.root / 'measurements.json', {'stage': self.stage})
                return {'inputs': [], 'human_review': ['review'], 'blockers': ['rule'],
                    'overall_status': status, 'provenance': {'qc_commands': []},
                    'checks': [{'rule_id': 'SEG.TISSUE', 'status': status, 'severity': 'error',
                        'message': 'measured', 'evidence': ['measurements.json'],
                        'metrics': [{'source': 'measurements.json'}], 'visual_evidence': []}]}

        with tempfile.TemporaryDirectory() as temporary, patch.object(qc_runtime, 'Evidence', FakeEvidence), \
                patch.object(qc_runtime, 'utc_now', return_value='fixed-time'):
            serial = qc_runtime.generate_group({'subject': 'sub-test'}, node,
                {'qc': {'threads': 1}}, Path(temporary) / 'serial')
            parallel = True
            concurrent = qc_runtime.generate_group({'subject': 'sub-test'}, node,
                {'qc': {'threads': 8}}, Path(temporary) / 'parallel')
            self.assertEqual(concurrent['provenance']['group_execution']['workers'], 2)
            self.assertEqual(concurrent['overall_status'], 'fail')
            for report in (serial, concurrent):
                report['provenance'].pop('group_execution')
            self.assertEqual(serial, concurrent)

    def test_worker_exception_does_not_publish_successful_group_report(self):
        node = {'node_id': 'group_qc', 'policy': {'checked_nodes': ['first', 'second']}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(qc_runtime, 'Evidence') as evidence:
            evidence.return_value.generate.side_effect = RuntimeError('measurement failed')
            with self.assertRaisesRegex(RuntimeError, 'measurement failed'):
                qc_runtime.generate_group({'subject': 'sub-test'}, node,
                    {'qc': {'threads': 2}}, temporary)
            self.assertFalse((Path(temporary) / 'group/report.json').exists())


if __name__ == '__main__':
    unittest.main()
