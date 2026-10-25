"""Re-ground a concept abstraction in a new visual domain, with the rule over it held fixed.

`regrounding.experiment.run` trains on the source style, re-grounds on ``k`` target scenes
and returns counts; `tools/study.py` runs it for every seed and mode and writes the study.
"""
