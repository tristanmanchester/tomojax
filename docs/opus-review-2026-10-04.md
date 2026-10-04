# Opus 5.5 review after the exact-projector rerun

**Recommendation: replace the fixed-volume pose step in `tomojax.align` with a variable-projection (Schur-complement) Gauss–Newton pose step.** Each outer iteration should compute the pose update from the joint linearised (volume, pose) least-squares problem, so the step accounts for how the volume will re-adapt. The existing nonnegative Huber-FISTA volume refresh stays as it is.

## Sources: the user's own messages

I read one file: `~/.codex/sessions/2026/10/03/rollout-2026-10-03T06-07-32-01a10029-0b42-7e01-a254-4032bc6e6787.jsonl`. It is the only session containing either ID, its cwd is `/home/tristan/Projects/tomojax`, and it carries the provider handoff from T3 thread `60146a03…`. I parsed the user-role `response_item` messages (timestamps are UTC) and set aside the injected `<codex_internal_context source="goal">` restatements.

- **Imported first message.** Make TomoJAX "a genuine option, not just a hobby project" alongside ASTRA/TIGRE.
- **05:07 and 05:13 on 10-03.** "If you were to set yourself an unrealistic goal…" and then "Set yourself the goal of reaching these targets". The stretch targets in `docs/optimization-goal.md` were the agent's own proposal, which the user endorsed. The user did not write them.
- **04:19 on 10-04.** Improve the whole system together. Extend methods to tilted/irregular geometry first. Keep the gates fixed. "If the change helps only a subset, say so and stop." For alignment: "time the real alignment path on a free-voxel object. Record the successful-recovery baseline… then reduce that time."
- **04:23 on 10-04.** "not focus on specific cases and try to 'beat' ASTRA… We're not producing some bench-maxxed slop… a real imaging library to be used by scientists."
- **07:18 on 10-04.** Commission this review; no new benchmark infrastructure.

**Where the goal restatement and the user's words disagree, I follow the user.** The goal doc frames joint recovery as "20× faster than the recorded baseline" and reconstruction as 10× over ASTRA. The 04:23 message overrides that ASTRA-ratio framing. On alignment, the user's 04:19 order is to reach success first and speed it up second. The 20× denominator does not exist yet: 0 of 48 calls succeeded. For a library scientists can use, joint alignment that fails at ±0.25° / ±0.5 px motion is a broken core capability, not a speed gap. Reconstruction is the opposite case: 26/27 cells are accepted and it is already usable. The ASTRA-ratio metric should not pull effort away from that.

## Evidence: one shared limiter in all six cells

I read every outer iteration in `public-free-voxel-v1-exact.json`:

- **Rotation is the only failing gate.**
  - Translation error is 0.0005–0.0027 px, roughly 20–100× under the 0.05 px gate.
  - Image error is 0.006–0.008 (gate 0.10) in parallel/anisotropic and 0.121 (gate 0.20) in laminography.
  - Rotation error is 0.022°, 0.122° and 0.219° against a 0.01° gate.
- **Anisotropic and laminography are still descending linearly at outer 64.**
  - Anisotropic rotation shrinks by about 0.80× every 8 outers (0.502 → 0.365 → 0.286 → 0.233 → 0.188 → 0.151 → 0.122).
  - Laminography shrinks by about 0.87× every 8 outers (0.506 → … → 0.219).
  - Extrapolating, anisotropic needs roughly 90 more outers and laminography roughly 150.
  - Each GN step reduces the loss by only about 3% (e.g. `loss_rel_pct −2.8`), and the steps are tiny (`rot_mean ≈ 7e-5` rad) while the error is still 0.2°.
  - This is the textbook signature of block-coordinate alternation, where the volume absorbs pose error.
- **Parallel stalls instead.** Gauge-fitted rotation sits at 0.020–0.022° from outer 24 to outer 64 (non-monotone), while the loss keeps falling (0.013 → 0.009) and raw rotation drifts slowly.
- **Variability is not the problem.** All 8 calls per cell agree to 3–4 digits.
- **The exact projector removed modelling bias but not the convergence problem.**
  - Rotation floors fell 4.5× in parallel and 2.6× in anisotropic.
  - Laminography barely moved (0.246 → 0.219), so it is convergence-limited, not bias-limited.
- **Speeding up kernels would not help.**
  - After the roughly 7.5 s first outer (compilation), each outer costs about 0.61 s: about 0.44 s of Pallas FISTA and 0.18 s of GN.
  - Iteration count is the limiter. The code confirms the structure: `_run_align_outer_iteration` (`src/tomojax/align/_pose/_pose_loop.py:436`) runs 20 FISTA iterations, then one per-view GN step on a frozen volume (`objective_kind: fixed_volume`).

**Why this happens.** The fixed-volume GN uses Jᵀ_θJ_θ as the curvature. That overstates curvature in exactly the directions the free voxels can compensate for. The proper reduced Hessian is the Schur complement:

S = Jᵀ_θJ_θ − Jᵀ_θA(AᵀA + μI)⁻¹AᵀJ_θ

Because S is smaller along those coupled directions, the current step is too short there, and the alternation crawls. Tilted and anisotropic geometry are worse conditioned, so they crawl fastest-to-slowest in the order seen.

## Scope

- **The step.** Solve the linearised joint problem min ‖Aδv + J_θδθ − r‖² + damping by CGLS/LSQR on the stacked operator [A, J_θ].
  - Use the matched exact forward/adjoint already in place.
  - J_θ is block-diagonal by view. Apply it from the per-view columns the central stencil already produces, which keeps the stencil, damping and acceptance check unchanged so the comparison stays controlled.
  - Keep only δθ, pass it through the existing loss-acceptance guard, then let FISTA refresh the volume with positivity as now.
- **Where it lives.** This is a general algorithm change in the public `align` path: operator-agnostic (sampled or exact) and geometry-agnostic, threaded through multiresolution like the integrator choice. It is not a fixture-specific tweak.
- **Cost.** About 20–40 extra forward/adjoint pairs per outer. Expect roughly 1–2 s per outer instead of 0.6 s, which pays off if outers drop from more than 64 to around 10.
- **Expected to benefit.** All per-view free-voxel alignment, with tilted/laminography and anisotropic/off-centre geometry gaining most.
- **Unaffected.**
  - The frozen 27-cell reconstruction matrix, including the failing sharp anisotropic-64 cell.
  - Cold compilation and setup cost.
  - The ±3° / ±10 px capture range and the 99% robustness distribution.

## How to check it with what already exists

1. **Correctness.** Extend the existing dense-matrix pose-normal tests to compare the matrix-free step against a dense Schur-complement solve on a tiny problem.
2. **Pilot rerun.** Rerun the unchanged six-cell pilot with the same gates, the 64×20 budget, the frozen fixtures and 1 cold + 7 warm calls. Report all 48 calls.
3. **Decision rule.** Success means the rotation gate is met in all six cells.
   - If only anisotropic and laminography recover and parallel stays at about 0.02°, report it as a subset result and stop this line, per the 04:19 instruction.
   - Any first successful run gives the successful-recovery baseline the user asked for. Time reductions come after that.

## Main uncertainty

The parallel plateau may not come from volume–pose coupling. It could instead be a floor from the central-difference Jacobian evaluated in FP32, or from the inexact, positivity-constrained FISTA inner solve. Either would cap rotation at about 0.02° whatever the curvature model.

I still chose this change for three reasons:
- Four of the six cells show unambiguous linear-rate alternation, and laminography shows the exact projector did not fix it.
- It is the one change that addresses the failure common to every geometry, rather than tuning a case.
- If parallel does not move while the other four cells converge, that result cleanly separates coupling from Jacobian precision. No new benchmark infrastructure is needed to tell them apart.
