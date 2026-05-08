# Rich Phantom Loss Comparison

| Case | Loss | det_u RMSE px | Volume NMSE | Final residual | Schur accepted | Classification |
|---|---|---:|---:|---:|---|---|
| setup_global_fixed_truth | otsu_l2 | 5.788025856018066 | 0.6814179420471191 | 74.88236236572266 | True | reconstruction_absorbed_geometry |
| setup_global_fixed_truth | pseudo_huber | 10.754942893981932 | 0.714363157749176 | 5.418478488922119 | True | reconstruction_absorbed_geometry |
| setup_global_fixed_truth | otsu_pseudo_huber | 11.009922742843626 | 0.7145728468894958 | 19.658485412597656 | True | training_loss_not_independent |
| setup_global_stopped | otsu_l2 | 0.8307647705078125 | 0.5306297540664673 | 55.30324172973633 | True | independent_projection_losses_consistent |
| setup_global_stopped | pseudo_huber | 9.645100116729736 | 0.7318097352981567 | 5.6957292556762695 | True | reconstruction_absorbed_geometry |
| setup_global_stopped | otsu_pseudo_huber | 4.143064498901367 | 0.7237358093261719 | 19.497346878051758 | True | reconstruction_absorbed_geometry |

Fixed-truth rows are oracle diagnostics, not production passes.
