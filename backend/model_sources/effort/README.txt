EFFORT EVALUATION SCRIPTS

Place evaluate_effort.py and tune_effort_threshold.py in:
the DeepfakeBench working directory containing the Effort source

Run all commands from that folder with the Effort .venv activated.

1) Validation inference (initial threshold 0.50; threshold does not affect scores):
   python evaluate_effort.py --split validation

2) Select operating threshold using validation only:
   python tune_effort_threshold.py

3) After the script prints the selected threshold, run independent test:
   python evaluate_effort.py --split test --threshold 0.XX

4) Run holdout with exactly the same frozen threshold:
   python evaluate_effort.py --split holdout --threshold 0.XX

Replace 0.XX with the threshold actually selected by the validation tuner.
Do not choose or adjust the threshold based on test or holdout metrics.

Outputs are written beside the scripts:
- effort_validation_results.csv
- effort_validation_frame_results.csv
- effort_validation_threshold_analysis.csv
- effort_test_results.csv
- effort_test_frame_results.csv
- effort_holdout_results.csv
- effort_holdout_frame_results.csv

Defaults:
- Manifest: the source-disjoint manifest configured for your local dataset
- Checkpoint: the Effort checkpoint configured for your local environment
- Config: training\config\detector\effort.yaml
- Local CLIP: models--openai--clip-vit-large-patch14
- Four evenly spaced PNG frames per video; video score is mean fake probability.
- Batch size 4 (reduce to 2 if CUDA runs out of memory).

Preprocessing follows the official demo when no landmark model is supplied:
BGR -> RGB -> 224x224 bilinear resize -> CLIP mean/std normalization.
No face-landmark crop is applied. This matches the no-landmark path in demo.py.

The script checks checkpoint key compatibility and permits only the expected fixed
SVD component keys (.S_r, .U_r, .V_r) to be absent. It stops for other missing
or unexpected keys. It does not modify or overwrite the checkpoint.

