#!/usr/bin/env julia
"""
Budget intuition plots — answering "is the uncertainty set meaningful?"

The uncertainty set is:
  U_s = { q : 0 ≤ q_od ≤ q̄_od,  Σ_od q_od ≤ Q̄_s }

"The budget does nothing" iff Σ q̄_od ≤ Q̄_s  (box fits inside cap → cap is slack).
"The budget is binding"    iff Σ q̄_od > Q̄_s  (adversary cannot push all OD pairs
                                              to their upper bounds simultaneously).

This script generates plots that make the budget's role concrete:

  fig12_budget_number_line.png   — For each period: show Σq̲, Q̄, Σq̄ on one axis.
                                   The "gap" Σq̄ − Q̄ is the adversary's constraint.

  fig13_adversary_budget.png     — Cumulative sum of sorted q̄_od values per period.
                                   The vertical mark at Q̄ shows how many OD pairs
                                   the adversary can simultaneously max out.

  fig14_demand_vs_cap.png        — Historical total demand per day (scatter),
                                   with Q̄ line. Shows what the cap actually cuts.

  fig15_worst_case_allocation.png — What the adversary actually does: greedy
                                   assignment of budget to highest-q̄ OD pairs.
                                   Compare to typical historical demand per OD pair.

Usage (from project root):
    julia scripts/analysis/analyze_budget_intuition.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "StationSelection.jl"))

using CSV, DataFrames, Dates, Statistics, JSON, Printf, Plots
gr(size=(900,600), dpi=150)

PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
bounds_file  = joinpath(PROJECT_ROOT, "Data", "demand_bounds.json")
order_file   = joinpath(PROJECT_ROOT, "Data", "zhuzhou_data", "order.csv")
out_dir      = joinpath(PROJECT_ROOT, "experiments", "demand_understanding")

const PERIODS = [
    (label="morning",   lo=6,  hi=10),
    (label="afternoon", lo=10, hi=15),
    (label="evening",   lo=15, hi=20),
    (label="night",     lo=20, hi=24),
]
const COLORS = [:steelblue, :darkorange, :forestgreen, :mediumpurple]

# ── Load bounds ───────────────────────────────────────────────────────────────
println("Loading bounds ...")
bounds_raw = JSON.parsefile(bounds_file)

function _parse_period_bounds(raw_period)
    Dict((parse(Int, split(k,",")[1]), parse(Int, split(k,",")[2])) => Float64(v)
         for (k,v) in raw_period)
end

q_low_all = [_parse_period_bounds(bounds_raw["q_low"][string(s)]) for s in 1:4]
q_hat_all = [_parse_period_bounds(bounds_raw["q_hat"][string(s)]) for s in 1:4]
Q_cap     = Float64.(bounds_raw["Q_cap"])
B         = Float64.(bounds_raw["B"])

# q̄_od = q̲_od + q̂_od
q_bar_all = [Dict(od => q_low_all[s][od] + q_hat_all[s][od]
                  for od in keys(q_hat_all[s])) for s in 1:4]

sum_q_low  = [sum(values(q_low_all[s])) for s in 1:4]
sum_q_bar  = [sum(values(q_bar_all[s])) for s in 1:4]
budget_gap = sum_q_bar .- Q_cap   # positive ⟹ cap is binding

# ── Load order data ───────────────────────────────────────────────────────────
println("Loading orders ...")
orders = CSV.read(order_file, DataFrame)
orders.order_dt   = DateTime.(orders.order_time, dateformat"yyyy-mm-dd HH:MM:SS")
orders.order_date = Date.(orders.order_dt)
orders.order_hour = hour.(orders.order_dt)
orders.period_idx = [let p = findfirst(p -> p.lo <= h < p.hi, PERIODS)
                       isnothing(p) ? 0 : p end for h in orders.order_hour]
orders_filt = filter(r -> r.period_idx > 0, orders)

gdf_total = combine(groupby(orders_filt, [:order_date, :period_idx]),
                    nrow => :total_demand)

# =============================================================================
# Figure 12: Budget number line — Σq̲, Q̄, Σq̄ for each period
# The gap between Q̄ and Σq̄ is what the budget "removes" from the box
# =============================================================================
println("Figure 12: budget number line ...")

p12 = plot(size=(820, 420), dpi=150,
    title="Uncertainty set: does the total-demand cap restrict the box?\n" *
          "(Binding iff Q̄ < Σq̄)",
    xlabel="total trips / day",
    yticks=(1:4, [pd.label for pd in PERIODS]),
    yflip=true,
    legend=:topright)

for s in 1:4
    y = s
    sq̲ = sum_q_low[s]
    Qc = Q_cap[s]
    sq̄ = sum_q_bar[s]

    # Span from 0 to Σq̄: show as a segment
    # Box region [0, Σq̄] in light color
    plot!(p12, [0, sq̄], [y, y]; lw=8, color=COLORS[s], alpha=0.25, label=nothing)
    # Feasible cap region [0, Q̄] in solid color
    plot!(p12, [0, Qc], [y, y]; lw=8, color=COLORS[s], alpha=0.8, label=nothing)

    # Markers
    scatter!(p12, [Qc], [y]; ms=10, color=:red, markershape=:vline, label=s==1 ? "Q̄  (cap)" : nothing)
    scatter!(p12, [sq̄], [y]; ms=10, color=:black, markershape=:vline, label=s==1 ? "Σq̄ (box max)" : nothing)
    scatter!(p12, [sq̲], [y]; ms=6, color=:white, markershape=:circle,
             markerstrokecolor=:black, markerstrokewidth=1.5,
             label=s==1 ? "Σq̲ (box min)" : nothing)

    # Annotate the gap
    annotate!(p12, (sq̄ + Qc)/2, y - 0.35,
        text("gap=$(round(Int, sq̄-Qc)) | ratio=$(round(budget_gap[s]/Qc*100; digits=0))% cut",
             7, :center, :gray40))
end

savefig(p12, joinpath(out_dir, "fig12_budget_number_line.png"))
println("  Saved fig12")

# =============================================================================
# Figure 13: Adversary budget — cumulative sorted q̄
# Shows how many OD pairs the adversary can simultaneously push to their max
# before exhausting the budget Q̄
# =============================================================================
println("Figure 13: adversary budget allocation ...")

p13 = plot(layout=(2,2), size=(1000,700), dpi=150,
    plot_title="Adversary's budget: cumulative Σq̄ when pushing OD pairs to their max\n" *
               "(vertical line = Q̄ — adversary must stop here)")

for s in 1:4
    # Only OD pairs with q̄ > 0
    q_bars = sort([v for v in values(q_bar_all[s]) if v > 0]; rev=true)
    n = length(q_bars)
    cum_qbar = cumsum(q_bars)

    # Find where cumulative sum first crosses Q̄
    k_exhausted = findfirst(>=(Q_cap[s]), cum_qbar)
    frac_exhausted = isnothing(k_exhausted) ? 1.0 : (k_exhausted / n)

    plot!(p13[s], 1:n, cum_qbar;
        lw=2, color=COLORS[s], label=nothing,
        xlabel="# OD pairs pushed to q̄  (sorted by q̄ desc.)",
        ylabel="cumulative Σq̄",
        title="$(PERIODS[s].label)  ($(n) active pairs)",
        titlefont=9)
    hline!(p13[s], [Q_cap[s]]; color=:red, lw=2, ls=:dash,
           label="Q̄ = $(round(Int, Q_cap[s]))")
    if !isnothing(k_exhausted)
        vline!(p13[s], [k_exhausted]; color=:black, lw=1.5, ls=:dot,
               label="budget exhausted at pair #$(k_exhausted) ($(round(100*frac_exhausted;digits=0))%)")
    end
    plot!(p13[s]; legend=:bottomright, legendfontsize=6)
end
savefig(p13, joinpath(out_dir, "fig13_adversary_budget.png"))
println("  Saved fig13")

# =============================================================================
# Figure 14: Historical total demand scatter vs cap
# Shows where Q̄ sits relative to daily realised demand
# =============================================================================
println("Figure 14: historical demand vs cap ...")

p14 = plot(layout=(2,2), size=(1100,700), dpi=150,
    plot_title="Historical total demand per day vs calibrated cap Q̄\n" *
               "(dots above red line = days outside the uncertainty set)")

for s in 1:4
    sub = sort(filter(r -> r.period_idx == s, gdf_total), :order_date)
    dates_idx = 1:nrow(sub)

    n_above   = count(>(Q_cap[s]), sub.total_demand)
    n_above_box = count(>(sum_q_bar[s]), sub.total_demand)  # above even the box max
    pct_above = round(100*n_above/nrow(sub); digits=1)

    scatter!(p14[s], dates_idx, sub.total_demand;
        ms=3, alpha=0.6, color=COLORS[s], label="daily total demand",
        xlabel="day index (sorted by date)",
        ylabel="total trips",
        title="$(PERIODS[s].label)  ($(pct_above)% days exceed Q̄)",
        titlefont=9)
    hline!(p14[s], [Q_cap[s]]; color=:red, lw=2, ls=:dash,
           label="Q̄ = $(round(Int,Q_cap[s]))  (cap)")
    hline!(p14[s], [sum_q_bar[s]]; color=:black, lw=1.5, ls=:dot,
           label="Σq̄ = $(round(Int,sum_q_bar[s]))  (box max)")
    hline!(p14[s], [sum_q_low[s]]; color=:gray, lw=1, ls=:solid,
           label="Σq̲ = $(round(Int,sum_q_low[s]))  (box min)")
    plot!(p14[s]; legend=:topright, legendfontsize=6)
end
savefig(p14, joinpath(out_dir, "fig14_demand_vs_cap.png"))
println("  Saved fig14")

# =============================================================================
# Figure 15: Worst-case vs typical demand per OD pair
# Adversary greedy: push highest-q̄ pairs to q̄, remaining budget to next pairs
# Typical: use per-OD median demand from historical data
# =============================================================================
println("Figure 15: worst-case vs typical OD allocation ...")

# Per-OD median demand (from historical data) for each period
gdf_od = combine(
    groupby(orders_filt, [:order_date, :period_idx,
                           :origin_station_id, :destination_station_id]),
    nrow => :demand,
)

p15 = plot(layout=(2,2), size=(1100,700), dpi=150,
    plot_title="Adversary worst-case vs. typical demand allocation\n" *
               "(top-30 OD pairs by q̄, sorted descending)")

for s in 1:4
    sub_hist = filter(r -> r.period_idx == s, gdf_od)
    od_med = combine(groupby(sub_hist, [:origin_station_id, :destination_station_id]),
                     :demand => median => :med_demand)

    # Build top-30 OD pairs by q̄
    active_ods = [(od, q_bar_all[s][od])
                  for od in keys(q_bar_all[s]) if q_bar_all[s][od] > 0]
    sort!(active_ods; by=x->x[2], rev=true)
    top_k = min(30, length(active_ods))
    top_ods = [active_ods[i][1] for i in 1:top_k]
    q_bar_top = [active_ods[i][2] for i in 1:top_k]

    # Adversary greedy allocation
    remaining = Q_cap[s] - sum_q_low[s]   # = B_s
    wc_demand = zeros(top_k)
    for i in 1:top_k
        ql = q_low_all[s][top_ods[i]]
        qh = q_hat_all[s][top_ods[i]]
        alloc = min(qh, remaining)
        wc_demand[i] = ql + alloc
        remaining -= alloc
        remaining <= 0 && break
    end

    # Typical demand (median) for these OD pairs
    med_lookup = Dict(
        (r.origin_station_id, r.destination_station_id) => r.med_demand
        for r in eachrow(od_med))
    typical_demand = [get(med_lookup, top_ods[i], 0.0) for i in 1:top_k]

    xs = 1:top_k
    bar!(p15[s], xs .- 0.22, wc_demand;
        bar_width=0.4, color=:tomato, alpha=0.75,
        label="worst-case (greedy budget alloc.)",
        xlabel="OD pair rank (by q̄ desc.)",
        ylabel="demand / day",
        title="$(PERIODS[s].label)",
        titlefont=9)
    bar!(p15[s], xs .+ 0.22, typical_demand;
        bar_width=0.4, color=COLORS[s], alpha=0.75,
        label="typical (median historical)")

    # Total demand annotations
    total_wc  = round(Int, sum(wc_demand))
    total_typ = round(Int, sum(typical_demand))
    annotate!(p15[s], top_k*0.55, maximum(q_bar_top)*0.92,
        text("WC total=$total_wc  Typical total=$total_typ", 6, :left))
    plot!(p15[s]; legend=:topright, legendfontsize=6)
end
savefig(p15, joinpath(out_dir, "fig15_worst_case_vs_typical.png"))
println("  Saved fig15")

# =============================================================================
# Print budget summary
# =============================================================================
println()
println("="^70)
println("BUDGET MEANINGFULNESS SUMMARY")
println("="^70)
println()
println("Key question: is Σq̄ > Q̄ ?  (Yes = cap is binding, uncertainty set is non-trivial)")
println()
@printf("%-12s  %6s  %6s  %6s  %8s  %8s  %8s\n",
    "Period", "Σq̲", "Q̄", "Σq̄", "gap=Σq̄-Q̄", "ratio", "cap at Q90?")
println("-"^70)
for s in 1:4
    sub = filter(r -> r.period_idx == s, gdf_total)
    pct_above = round(100 * mean(sub.total_demand .> Q_cap[s]); digits=1)
    @printf("%-12s  %6.1f  %6.1f  %6.1f  %+9.1f  %8.2f  %s%% days above\n",
        PERIODS[s].label, sum_q_low[s], Q_cap[s], sum_q_bar[s],
        budget_gap[s], sum_q_bar[s]/Q_cap[s], pct_above)
end
println()
println("Interpretation:")
println("  - ratio > 1  →  cap IS binding (not pathological)")
println("  - gap = Σq̄ − Q̄ = demand the adversary 'cannot use' due to the cap")
println("  - ~10% days above cap confirms Q90 calibration is working correctly")
println()
println("All figures saved to: $out_dir")
println("="^70)
