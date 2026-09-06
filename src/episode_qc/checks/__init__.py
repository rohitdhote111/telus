"""Individual check modules. Each exposes a ``run(episode, calibration, ctx)``
function returning a list of ``episode_qc.scoring.Finding``. ``ctx`` is a
small object carrying corpus-wide values (populated once per run in
``cli.py``) so a check can compare one episode/camera against the rest of
the corpus rather than a fixed constant.
"""
