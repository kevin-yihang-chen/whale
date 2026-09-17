"""Tests for deterministic recovery of a completed, paid proposal."""

import ast
import unittest

from ours.recover_scoped_proposal import normalize_candidate_source


class CandidateSourceNormalizationTests(unittest.TestCase):
    def test_normalizes_only_local_private_names_and_integer_turn_count(self):
        source = '''
_PROMPT = "keep _PROMPT and len(assistant_turns) verbatim"
def nudge(text, assistant_turns):
    _count = len(assistant_turns)
    return _PROMPT if _count < 2 else None
'''
        normalized, mapping, replacements = normalize_candidate_source(source)

        self.assertEqual(mapping, {"_PROMPT": "PROMPT", "_count": "count"})
        self.assertEqual(replacements, 1)
        self.assertIn('"keep _PROMPT and len(assistant_turns) verbatim"', normalized)
        self.assertIn("count = assistant_turns", normalized)
        self.assertFalse(any(
            isinstance(node, ast.Name) and node.id.startswith("_")
            for node in ast.walk(ast.parse(normalized))
        ))

    def test_rejects_private_attribute_access(self):
        with self.assertRaisesRegex(ValueError, "Private attribute access"):
            normalize_candidate_source("def nudge(text, assistant_turns):\n    return text._secret\n")

    def test_rejects_public_name_collision(self):
        source = "VALUE = 1\n_VALUE = 2\n"
        with self.assertRaisesRegex(ValueError, "collide"):
            normalize_candidate_source(source)

    def test_rejects_python_special_identifiers(self):
        with self.assertRaisesRegex(ValueError, "special identifiers"):
            normalize_candidate_source("if __name__ == '__main__':\n    pass\n")

    def test_does_not_rewrite_other_len_calls(self):
        source = '''
def nudge(text, assistant_turns):
    return str(len(text)) + str(assistant_turns)
'''
        normalized, mapping, replacements = normalize_candidate_source(source)
        self.assertEqual(mapping, {})
        self.assertEqual(replacements, 0)
        self.assertIn("len(text)", normalized)


if __name__ == "__main__":
    unittest.main()
