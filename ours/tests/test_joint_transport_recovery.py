"""Transport retry cannot reset the original paid request ceiling."""
import unittest

from ours.joint_transport_recovery import available_requests, provider_errors


class TransportRecoveryTests(unittest.TestCase):
    def test_provider_error_done_marker_is_not_a_successful_usage_receipt(self):
        raw = 'event: error\ndata: {"type":"error","error":{"message":"Internal Network Failure"}}\n\ndata: [DONE]\n'
        self.assertEqual(provider_errors(raw), [{'message': 'Internal Network Failure'}])
        with self.assertRaises(ValueError):
            provider_errors('data: invalid JSON\n')

    def test_failed_requests_count_and_final_round_is_reserved(self):
        ids = [str(i) for i in range(100)]
        snapshot = {'ledger_request_ids': ids, 'reservations_before_search': 62}
        self.assertEqual(available_requests(snapshot, 4, ids), 10)
        self.assertEqual(available_requests(snapshot, 5, ids + ['new'+str(i) for i in range(10)]), 12)
        with self.assertRaises(ValueError):
            available_requests(snapshot, 5, ids + ['new'+str(i) for i in range(22)])
        with self.assertRaises(ValueError):
            available_requests(snapshot, 4, ['changed'] + ids[1:])
        with self.assertRaises(ValueError):
            available_requests(snapshot, 3, ids)


if __name__ == '__main__':
    unittest.main()
