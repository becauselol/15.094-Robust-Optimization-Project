#!/usr/bin/env julia
"""
Calibrate demand bounds from historical zhuzhou_data for the robust model.

Lower bounds are fixed at zero (q̲ = 0 for all OD pairs).  See
StationSelection.jl/src/utils/data/demand_bounds.jl for the rationale.

Usage (from project root):
    julia scripts/generate_demand_bounds.jl [--q-high 0.90] [--q-cap 0.90]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "StationSelection.jl"))
using StationSelection
using Dates, JSON, ArgParse

function parse_args_local()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--q-high"
            help    = "Upper quantile for per-OD demand bounds"
            default = 0.90
            arg_type = Float64
        "--q-cap"
            help    = "Quantile for total-demand cap Q̄_s"
            default = 0.90
            arg_type = Float64
        "--start-date"
            help    = "Start date for historical data (YYYY-MM-DD)"
            default = "2025-01-01"
            arg_type = String
        "--end-date"
            help    = "End date for historical data (YYYY-MM-DD)"
            default = "2025-06-30"
            arg_type = String
    end
    return parse_args(s)
end

args = parse_args_local()

PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
station_file = joinpath(PROJECT_ROOT, "Data", "zhuzhou_data", "station.csv")
order_file   = joinpath(PROJECT_ROOT, "Data", "zhuzhou_data", "order.csv")
segment_file = joinpath(PROJECT_ROOT, "Data", "zhuzhou_data", "segment.csv")
output_file  = joinpath(PROJECT_ROOT, "Data", "demand_bounds.json")

println("="^80)
println("Demand Bound Calibration")
println("="^80)
println("q̲ = 0 (fixed)  q_high=$(args["q-high"])  q_cap=$(args["q-cap"])")
println("Date range: $(args["start-date"]) → $(args["end-date"])")
println()

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading stations and orders...")
stations = read_candidate_stations(station_file)
requests = read_customer_requests(order_file)

walking_costs = compute_station_pairwise_costs(stations)
routing_costs = read_routing_costs_from_segments(segment_file, stations)

# ── Generate one ScenarioData per (date × period) over the full date range ───
println("Generating historical scenarios (four_period profile)...")
start_date = Date(args["start-date"])
end_date   = Date(args["end-date"])

scenario_ranges = generate_scenarios_by_profile(start_date, end_date; profile=:four_period)
println("  Generated $(length(scenario_ranges)) scenario instances")

data = create_station_selection_data(
    stations, requests, walking_costs;
    routing_costs=routing_costs, scenarios=scenario_ranges
)

println("  Scenarios loaded: $(n_scenarios(data))")

# ── Group by period and calibrate ─────────────────────────────────────────────
println("Grouping scenarios by period and computing bounds...")
groups = group_scenarios_by_period(data.scenarios)
for (period, scs) in sort(collect(groups); by=first)
    labels = ["morning", "afternoon", "evening", "night"]
    println("  Period $(labels[period]): $(length(scs)) historical instances")
end

q_low, q_hat, B, Q_cap = compute_demand_bounds(
    groups;
    q_high_quantile = args["q-high"],
    Q_cap_quantile  = args["q-cap"],
)

# ── Serialise ─────────────────────────────────────────────────────────────────
function _dict_to_json(d::Dict{Int, Dict{Tuple{Int,Int}, Float64}})
    Dict(string(s) => Dict("$(od[1]),$(od[2])" => v for (od, v) in od_dict)
         for (s, od_dict) in d)
end

output = Dict(
    "q_low"       => _dict_to_json(q_low),
    "q_hat"       => _dict_to_json(q_hat),
    "B"           => B,
    "Q_cap"       => Q_cap,
    "calibration" => Dict(
        "q_low"           => 0.0,
        "q_high_quantile" => args["q-high"],
        "Q_cap_quantile"  => args["q-cap"],
        "start_date"      => args["start-date"],
        "end_date"        => args["end-date"],
        "n_scenarios"     => length(scenario_ranges),
    ),
    "periods" => Dict(
        "1" => "morning (06-10)",
        "2" => "afternoon (10-15)",
        "3" => "evening (15-20)",
        "4" => "night (20-24)",
    ),
)

open(output_file, "w") do f
    JSON.print(f, output, 2)
end

println()
println("="^80)
println("B (demand budget per period):")
labels = ["morning", "afternoon", "evening", "night"]
for s in 1:4
    println("  $(labels[s]):  Q_cap=$(round(Q_cap[s]; digits=1))  B=$(round(B[s]; digits=1))  " *
            "n_od_pairs=$(length(q_low[s]))")
end
println()
println("✓ Saved: $output_file")
println("="^80)
