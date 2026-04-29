#!/usr/bin/env julia
"""
Visualize demand distributions across the 4 scenario time periods.

Generates 5 figures saved to experiments/demand_understanding/:
  fig1_total_demand_distributions.png  — histogram of total daily demand per period
  fig2_od_demand_distributions.png     — per-OD demand spread (violin / box plots)
  fig3_od_pair_counts.png              — number of unique OD pairs per period per day
  fig4_total_demand_cdf.png            — empirical CDF with Q_cap percentile markers
  fig5_od_demand_heatmap.png           — mean demand by OD pair (top pairs shown)

Usage (from project root):
    julia scripts/analysis/visualize_demand.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "StationSelection.jl"))

using CSV, DataFrames, Dates, Statistics, JSON, Printf

# Load Plots lazily (install if missing)
try
    using Plots
catch
    Pkg.add("Plots")
    using Plots
end
gr(size=(900,600), dpi=150)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
order_file   = joinpath(PROJECT_ROOT, "Data", "zhuzhou_data", "order.csv")
out_dir      = joinpath(PROJECT_ROOT, "experiments", "demand_understanding")
mkpath(out_dir)

# ── Period definitions (matches :four_period profile) ─────────────────────────
const PERIODS = [
    (label="morning",   lo=6,  hi=10),
    (label="afternoon", lo=10, hi=15),
    (label="evening",   lo=15, hi=20),
    (label="night",     lo=20, hi=24),
]
const COLORS = [:steelblue, :darkorange, :forestgreen, :mediumpurple]

function period_of(h::Int)
    for (i, p) in enumerate(PERIODS)
        p.lo <= h < p.hi && return i
    end
    return nothing  # e.g. 00-06 early morning — excluded
end

# ── Load orders ───────────────────────────────────────────────────────────────
println("Loading orders from $order_file ...")
orders = CSV.read(order_file, DataFrame)
println("  Loaded $(nrow(orders)) orders")

# Parse order_time and add helper columns
orders.order_dt   = DateTime.(orders.order_time, dateformat"yyyy-mm-dd HH:MM:SS")
orders.order_date = Date.(orders.order_dt)
orders.order_hour = hour.(orders.order_dt)
orders.period_idx = [period_of(h) for h in orders.order_hour]

# Drop early-morning trips (00-06) — outside all 4 windows
orders_filtered = filter(r -> !isnothing(r.period_idx), orders)
println("  Orders within 4 periods: $(nrow(orders_filtered)) " *
        "($(nrow(orders) - nrow(orders_filtered)) dropped, outside 06-24)")

# ── Aggregate: total demand per (date, period) ────────────────────────────────
gdf_total = combine(
    groupby(orders_filtered, [:order_date, :period_idx]),
    nrow => :total_demand,
    :pax_num => sum => :total_pax,
)
gdf_total.period_label = [PERIODS[i].label for i in gdf_total.period_idx]

# ── Aggregate: per-OD-pair demand per (date, period) ─────────────────────────
gdf_od = combine(
    groupby(orders_filtered, [:order_date, :period_idx,
                               :origin_station_id, :destination_station_id]),
    nrow => :od_demand,
)
gdf_od.od_label = string.(gdf_od.origin_station_id, "→", gdf_od.destination_station_id)

println("\nSummary by period:")
for i in 1:4
    sub = filter(r -> r.period_idx == i, gdf_total)
    println("  $(PERIODS[i].label): $(nrow(sub)) days | " *
            "mean=$(round(mean(sub.total_demand); digits=1)) | " *
            "p90=$(round(quantile(sub.total_demand, 0.90); digits=1)) | " *
            "max=$(maximum(sub.total_demand))")
end
println()

# =============================================================================
# Figure 1: Total daily demand distributions (histogram per period, 4 panels)
# =============================================================================
println("Generating Figure 1: total demand distributions...")
p1 = plot(layout=(2,2), size=(1000,700), dpi=150,
          plot_title="Total daily demand per period")

for (i, pd) in enumerate(PERIODS)
    sub   = filter(r -> r.period_idx == i, gdf_total).total_demand
    q90   = quantile(sub, 0.90)
    q95   = quantile(sub, 0.95)
    q99   = quantile(sub, 0.99)
    histogram!(p1[i], sub;
        bins       = 25,
        color      = COLORS[i],
        alpha      = 0.75,
        label      = nothing,
        xlabel     = "trips / day",
        ylabel     = "frequency",
        title      = "$(pd.label) ($(pd.lo):00–$(pd.hi):00)",
        titlefont  = 9,
    )
    vline!(p1[i], [q90]; color=:red,    lw=1.5, ls=:dash,  label="Q90=$(round(Int,q90))")
    vline!(p1[i], [q95]; color=:orange, lw=1.5, ls=:dot,   label="Q95=$(round(Int,q95))")
    vline!(p1[i], [q99]; color=:black,  lw=1.5, ls=:solid, label="Q99=$(round(Int,q99))")
    plot!(p1[i]; legend=:topright, legendfontsize=7)
end
savefig(p1, joinpath(out_dir, "fig1_total_demand_distributions.png"))
println("  Saved fig1")

# =============================================================================
# Figure 2: Per-OD demand spread — box plots of (date, period) OD counts
# (top 30 most common OD pairs per period to keep readable)
# =============================================================================
println("Generating Figure 2: per-OD demand distributions...")
p2 = plot(layout=(2,2), size=(1200,700), dpi=150,
          plot_title="Per-OD-pair daily demand (top-30 OD pairs)")

for (i, pd) in enumerate(PERIODS)
    sub = filter(r -> r.period_idx == i, gdf_od)

    # Rank OD pairs by mean demand
    od_means = combine(groupby(sub, :od_label), :od_demand => mean => :mean_demand)
    sort!(od_means, :mean_demand; rev=true)
    top_ods = od_means.od_label[1:min(30, nrow(od_means))]

    sub_top = filter(r -> r.od_label in top_ods, sub)

    # Compute mean ± std per top OD pair
    means_od = [mean(filter(r -> r.od_label == od, sub_top).od_demand) for od in top_ods]
    stds_od  = [std( filter(r -> r.od_label == od, sub_top).od_demand) for od in top_ods]

    bar!(p2[i], 1:length(top_ods), means_od;
        yerror     = stds_od,
        color      = COLORS[i],
        alpha      = 0.7,
        label      = nothing,
        xticks     = (1:length(top_ods), top_ods),
        xrotation  = 60,
        xlabel     = "",
        ylabel     = "trips / day (mean ± σ)",
        title      = "$(pd.label) — top $(length(top_ods)) OD pairs",
        titlefont  = 9,
    )
    plot!(p2[i]; bottom_margin=10Plots.mm, xtickfontsize=5)
end
savefig(p2, joinpath(out_dir, "fig2_od_demand_distributions.png"))
println("  Saved fig2")

# =============================================================================
# Figure 3: Number of active OD pairs per day per period
# =============================================================================
println("Generating Figure 3: active OD pair counts...")
gdf_od_count = combine(
    groupby(gdf_od, [:order_date, :period_idx]),
    nrow => :n_od_pairs,
)

p3 = plot(layout=(2,2), size=(1000,700), dpi=150,
          plot_title="Active OD pairs per day per period")

for (i, pd) in enumerate(PERIODS)
    sub = filter(r -> r.period_idx == i, gdf_od_count).n_od_pairs
    histogram!(p3[i], sub;
        bins       = 20,
        color      = COLORS[i],
        alpha      = 0.75,
        label      = nothing,
        xlabel     = "unique OD pairs",
        ylabel     = "frequency",
        title      = "$(pd.label)  median=$(round(Int, median(sub)))",
        titlefont  = 9,
    )
    vline!(p3[i], [median(sub)]; color=:black, lw=2, ls=:dash, label="median")
    plot!(p3[i]; legend=:topright, legendfontsize=7)
end
savefig(p3, joinpath(out_dir, "fig3_od_pair_counts.png"))
println("  Saved fig3")

# =============================================================================
# Figure 4: Empirical CDF of total demand — one curve per period
# with Q_cap percentile markers
# =============================================================================
println("Generating Figure 4: total demand CDF...")
p4 = plot(size=(800,500), dpi=150,
          xlabel="total trips / day",
          ylabel="empirical CDF",
          title="Total demand CDF — all 4 periods",
          legend=:bottomright)

for (i, pd) in enumerate(PERIODS)
    sub = sort(filter(r -> r.period_idx == i, gdf_total).total_demand)
    n   = length(sub)
    cdf = (1:n) ./ n
    plot!(p4, sub, cdf; color=COLORS[i], lw=2, label=pd.label)
    for (q, name) in [(0.90,"90"), (0.95,"95"), (0.99,"99")]
        qv = quantile(sub, q)
        scatter!(p4, [qv], [q]; color=COLORS[i], ms=5, label=nothing,
                 markershape=:diamond)
    end
end

# annotate percentile lines
for (q, name) in [(0.90,"Q90"), (0.95,"Q95"), (0.99,"Q99")]
    hline!(p4, [q]; color=:gray, lw=0.8, ls=:dot, label=name)
end
savefig(p4, joinpath(out_dir, "fig4_total_demand_cdf.png"))
println("  Saved fig4")

# =============================================================================
# Figure 5: Mean OD demand heatmap — top 25×25 origin/destination stations
# (one subplot per period)
# =============================================================================
println("Generating Figure 5: mean OD demand heatmap...")

# Find the top-25 stations by total order volume across all periods
station_vol = combine(
    groupby(orders_filtered, :origin_station_id),
    nrow => :vol
)
sort!(station_vol, :vol; rev=true)
top_stations = station_vol.origin_station_id[1:min(25, nrow(station_vol))]

p5 = plot(layout=(2,2), size=(1200,1000), dpi=150,
          plot_title="Mean daily OD demand — top-25 stations")

for (i, pd) in enumerate(PERIODS)
    sub = filter(r -> r.period_idx == i &&
                       r.origin_station_id in top_stations &&
                       r.destination_station_id in top_stations, gdf_od)

    # Build mean demand matrix
    n = length(top_stations)
    M = zeros(n, n)
    idx = Dict(s => j for (j, s) in enumerate(top_stations))

    od_means = combine(groupby(sub, [:origin_station_id, :destination_station_id]),
                       :od_demand => mean => :mean_demand)
    for row in eachrow(od_means)
        haskey(idx, row.origin_station_id) || continue
        haskey(idx, row.destination_station_id) || continue
        M[idx[row.origin_station_id], idx[row.destination_station_id]] = row.mean_demand
    end

    heatmap!(p5[i], M;
        color      = :YlOrRd,
        xlabel     = "destination station",
        ylabel     = "origin station",
        title      = "$(pd.label)",
        titlefont  = 9,
        colorbar_title = "mean trips/day",
    )
end
savefig(p5, joinpath(out_dir, "fig5_od_demand_heatmap.png"))
println("  Saved fig5")

# =============================================================================
# Print summary table: q_low (Q10), q_high (Q90), Q_cap (Q90) per period
# =============================================================================
println("\n" * "="^70)
println("Uncertainty Set Summary (illustrative — actual bounds from generate_demand_bounds.jl)")
println("="^70)
println("Period       | Q10 total | Q50 total | Q90 total | Q95 total | n_days")
println("-"^70)
for i in 1:4
    sub = filter(r -> r.period_idx == i, gdf_total).total_demand
    @printf("%-12s | %9.1f | %9.1f | %9.1f | %9.1f | %6d\n",
        PERIODS[i].label,
        quantile(sub, 0.10),
        quantile(sub, 0.50),
        quantile(sub, 0.90),
        quantile(sub, 0.95),
        length(sub),
    )
end
println()

println("Output files saved to: $out_dir")
println("="^70)
