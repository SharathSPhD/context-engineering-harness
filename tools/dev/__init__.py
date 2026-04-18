"""Dev-only tooling for the v2 rebuild.

Anything in this package is for orchestrating the rebuild itself
(scheduler, ralph-loop / attractor-flow integration, paper build).
NONE of it is shipped in the pratyaksha-context-eng-harness plugin.

A CI grep gate enforces this separation.
"""
