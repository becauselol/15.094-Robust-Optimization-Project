# Uncertainty Set Analysis Notes

Generated: 2026-04-27

## Model recap

Uncertainty set for period s:

    U_s = { q_s : q̲_od ≤ q_od ≤ q̄_od  ∀od,   Σ_od q_od ≤ Q̄_s }

where q̲ = Q₁₀, q̄ = Q₉₀ of zero-padded historical OD demand,
and Q̄_s = Q₉₀ of historical total daily demand for period s.

## 1. Is the total-demand cap binding? (Non-pathology check)

The cap is **non-binding** (pathological / degenerates to box) iff Σ q̄_od ≤ Q̄_s.

| Period    | Σq̲   | Σq̄   | Q̄    | ratio Σq̄/Q̄ |
|-----------|-------|-------|-------|------------|
| morning | 0.0 | 173.0 | 110.0 | 1.57 |
| afternoon | 2.0 | 503.7 | 376.9 | 1.34 |
| evening | 15.2 | 936.2 | 695.6 | 1.35 |
| night | 2.0 | 255.7 | 208.5 | 1.23 |

**All ratios > 1 → the cap is genuinely binding in every period.**

This means not all OD pairs can simultaneously reach their upper bounds;
the robust counterpart is strictly tighter than a pure box model.
Morning has the highest ratio (1.57×),
suggesting the cap cuts off the most combinatorial slack there.

## 2. Lower bounds are essentially zero (Σq̲ ≈ 0)

In every period Σq̲ ≈ 0.  This occurs because most OD pairs are **sporadic**:
they appear on fewer than ~10% of days, so their Q₁₀ = 0.
Consequence: the effective uncertainty set simplifies to

    U_s ≈ { q_s : 0 ≤ q_od ≤ q̂_od  ∀od,   Σ_od q_od ≤ Q̄_s }

and B_s ≈ Q̄_s.  The budget is not "consumed" by lower bounds at all.
This is not pathological — it just means the lower-bound terms in the objective
(Σ q̲ · t) are negligible, and robustness comes entirely from the B and q̂ terms.

## 3. Only a small fraction of OD pairs carry non-trivial upper bounds

| Period    | Total pairs | q̂ > 0 | Active fraction |
|-----------|-------------|--------|-----------------|
| morning | 901 | 102 | 11.3% |
| afternoon | 1589 | 221 | 13.9% |
| evening | 1802 | 319 | 17.7% |
| night | 851 | 121 | 14.2% |

A pair has q̂ > 0 iff it appears on **≥ 10% of historical days**.
The other 82–89% of pairs have q̂ = 0 — they are too sporadic to get an
upper bound and are excluded from the uncertainty set (treated as fixed-zero demand).

This has an important model consequence: those zero-bound pairs *always receive
zero demand in the robust worst case*, which may under-represent demand spikes
from rare but high-impact pairs.

## 4. Historical budget utilisation

The calibration target for Q̄ is Q₉₀ — so ~10% of days should violate the cap.
Measured violation rates match expectations well. The median utilisation
(Σq - Σq̲) / B varies by period, showing different levels of typical demand
relative to the cap.

## 5. Demand concentration (Lorenz curves)

Demand concentration (fig10) is high: a small fraction of "active" OD pairs
(those with q̂ > 0) account for the bulk of Σq̄.  This means:
- A few high-frequency corridor pairs drive most of the worst-case cost
- Most sporadic pairs contribute near-zero even in the worst case

## 6. Individual OD bound violations

Even among the active OD pairs, the Q₉₀ upper bounds are violated on roughly 10%
of days (by construction).  Fig11 shows this is consistent across periods.

## 7. Conclusions for model validity

✓ The uncertainty set is **non-degenerate**: the total-demand cap genuinely
  restricts the feasible demand region beyond what individual OD bounds do alone.

✓ The budget ratio (1.23–1.57×) is modest, meaning the set is not overly
  conservative; worst-case demand is not wildly above typical demand.

✗ Lower bounds are zero for almost all OD pairs.  The q̲-weighted terms in the
  robust objective vanish, leaving a simpler structure: the robust solution
  only needs to hedge against OD pairs with q̂ > 0 under the total-demand cap.

⚠ Sporadic pairs (presence < 10%) are invisible to the uncertainty set.
  If a rare OD pair with large demand spikes is misassigned, the model
  provides no guarantee.  A possible fix: use a smaller q-high quantile (e.g.,
  Q₇₅ or Q₈₀) so more pairs get non-zero upper bounds.
