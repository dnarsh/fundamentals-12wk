# Provenance

| Part | Author |
| --- | --- |
| Scaffold, specs (`README.md` per week), tests, CI, root docs | Written by an AI agent (Claude) |
| Every `solution.py` | Hand-written by the learner. **Empty so far**: stubs only, signatures and `raise NotImplementedError`. |
| `LEARNING-LOG.md` entries, "what I got wrong" sections | Written by the learner |

The agent wrote no solution code. Tests check properties and invariants (and a few hand-derived values), not a shipped reference implementation.

Caveat: the tests have been run against the stubs (all skipped) and against deliberately wrong stubs (all fail). They have not been run against a correct implementation, so a test bug is possible. If you think a test is wrong, open an issue with your reasoning before editing it.
