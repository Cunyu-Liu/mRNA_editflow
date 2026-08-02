# MK0 structural STOP semantics

## Boundary

Explicit STOP is an mRNA-EditFlow project extension, not an original Edit Flows action. MK0-v1 models a structural post-completion STOP process. It does not claim to know when continued editing would stop improving biological function; that question is reserved for a future FC0-versioned functional critic.

## Auxiliary survival construction

For changing aligned coordinates with independent switch clocks, define structural completion:

\[
\tau_{comp}=\max_{i\in\mathcal I_\Delta}\tau_i,
\]

with `tau_comp=0` for a zero-edit pair. Independently draw

\[
D\sim Exponential(\gamma_{ref}),\qquad D>0\text{ almost surely},
\]

and set

\[
\tau_{stop}=\tau_{comp}+D,\quad
\tilde\tau=\min(\tau_{stop},1),\quad
\delta=1[\tau_{stop}<1].
\]

The frozen primary uses `gamma_ref=16`; `8` and `32` are preregistered development sensitivities. Dwell is independent of all edit switch clocks. Positive dwell prevents a STOP atom at time zero for identity pairs and prevents STOP from sharing the last edit time.

The auxiliary state is ACTIVE before an observed STOP and HALTED afterward. An administratively censored path remains ACTIVE through horizon one. Once HALTED, sequence, mapping, history and budget freeze, edit actions have zero legality, and edit-flow target terms are absent.

## Absolute-hazard loss

Using the predictable pre-event state:

\[
\mathcal L_{stop}=
-\delta\log\lambda_{stop}(S_{\tilde\tau^-})+
\int_0^{\tilde\tau}1[q_s=ACTIVE]\lambda_{stop}(S_{s^-})ds.
\]

The integral uses 64-node Gauss-Legendre quadrature and a 128-node reference under frozen float64 tolerances. The network receives the explicit clock at every quadrature point.

A competing-risk ratio such as `lambda_stop/(lambda_stop+Lambda_edit)` is scale-invariant and cannot identify an absolute STOP intensity. It remains a clearly labelled non-CTMC event-type ablation and is not the MK0 primary.

## Termination state machine

| Reason | Learned event? | Forced? | Inside the CTMC? | Valid candidate? |
|---|---:|---:|---:|---:|
| `LEARNED_STOP` | yes | no | yes | yes, if state is otherwise valid |
| `FORCED_BUDGET` | no | yes | no | yes, but never a correct STOP prediction |
| `FORCED_NO_LEGAL_EDIT_ACTION` | no | yes | no | yes, with STOP hazard reported separately |
| `FORCED_ZERO_REMAINING_INTEGRATED_HAZARD` | no | yes | no | yes only after separate integral verification |
| `FORCED_TIME_HORIZON` | no | yes | no | yes, with never-STOP accounting |
| `FAILED_NUMERICAL` | no | no | no | no |

`NO_EVENT` means only that time advances without an event in one sampler substep. It is not STOP. Exhausted edit budget does not itself prove that the model predicted STOP. Forced inference termination is not treated as non-informative censoring in the learned auxiliary likelihood.

Three zero-hazard cases remain distinct:

1. zero edit hazard with positive STOP hazard;
2. zero instantaneous total hazard, which advances time and recomputes rates without division; and
3. zero remaining integrated total hazard, which permits the dedicated forced reason only after numerical verification.

## Oracles and reporting

The STOP artifact binds:

- analytic versus numerical survival likelihood for constant and piecewise hazards;
- sensitivity to absolute scaling of STOP rate;
- positivity and independence diagnostics for dwell draws;
- analytic versus Monte Carlo event-rate and administrative-censor fraction for gamma 8/16/32;
- quadrature error and endpoint handling;
- premature-STOP and never-STOP counts;
- learned/forced proportions and every termination reason;
- budget exhaustion, zero-instantaneous-hazard steps and separately verified zero-remaining-integral events; and
- the task's explicit step-0 identity-output policy (`false` in MK0-v1).

Failure of calibration, censor-fraction, numerical integration, premature-STOP or never-STOP checks is retained as evidence and blocks MK0 rather than changing the target definition.
