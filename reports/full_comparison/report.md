# Full Comparison Report

Generated from saved real-run CSV artifacts.

## Final Results

| label | env_id | final_step | final_success_rate | final_mean_reward | best_success_rate | best_success_step | steps_to_success_50 | auc_success_normalized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FetchReach SAC | FetchReachDense-v3 | 250000 | 1 | -0.8798 | 1 | 90000 | 3e+04 | 0.829 |
| FetchPushDense SAC | FetchPushDense-v3 | 500000 | 0.05 | -7.823 | 0.2 | 360000 | NA | 0.0625 |
| FetchPushSparse SAC+HER | FetchPush-v3 | 500000 | 0.9 | -15.65 | 0.9 | 450000 | 4e+05 | 0.226 |
| FetchReach JEPA-MPC | FetchReachDense-v3 | 0 | 0.35 | -3.656 | 0.35 | 0 | NA | 0.35 |

## JEPA Training

| epoch | train_loss | val_loss | latent_std_mean | latent_std_min | latent_norm_mean |
| --- | --- | --- | --- | --- | --- |
| 99 | 0.8415 | 0.8415 | 0.0869 | 0.06707 | 1 |

## Artifacts

- `final_metrics.csv`: summary table across methods/tasks.
- `learning_curve_points.csv`: all eval rows used in the plot.
- `jepa_final_train_metrics.csv`: final JEPA training row.
- `plots/success_curves.png`: combined success curve plot.
