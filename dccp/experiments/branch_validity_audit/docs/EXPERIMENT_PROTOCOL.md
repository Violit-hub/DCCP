# Frozen protocol checklist

1. Source trajectories are held out from VLA, WM, and LRM training.
2. Every state has complete MuJoCo state, model XML, RGB frame, nominal action, and hashes.
3. Candidate manifest is frozen once; WM and simulator read the same NPZ files.
4. WM mode and seeds are declared before scoring.
5. Every state × candidate × WM seed prediction has an inspectable MP4.
6. Every prediction receives a Progress LRM score.
7. `LABELS_FROZEN.json` verifies before any simulator rollout.
8. Every state × candidate × simulator seed outcome exists.
9. Simulator success and Coffee milestones—not LRM—define ground truth.
10. Report both all-pair strict and decisive-pair accuracy so real ties remain visible.
11. Inspect infrastructure failures; never convert them silently into task failures.
12. Archive the config, manifests, HTML page, CSVs, and JSON report together.

Use controlled noise as the primary result and independent (`faithful`) noise as robustness. Never combine modes under one `run_name`.
