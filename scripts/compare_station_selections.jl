#!/usr/bin/env julia

using CSV, DataFrames
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

function build_rows(exp_dir::String)
    rows = NamedTuple[]
    for (nominal, robust) in pair_runs(exp_dir)
        y_nom = parse_station_selection(String(nominal["run_dir"]))
        y_rob = parse_station_selection(String(robust["run_dir"]))
        z_nom = parse_scenario_activation(String(nominal["run_dir"]))
        z_rob = parse_scenario_activation(String(robust["run_dir"]))
        push!(rows, (
            k = nominal["k"],
            in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
            quantile = robust["comparison_quantile"],
            nominal_job_id = nominal["job_id"],
            robust_job_id = robust["job_id"],
            y_j_exact_same = y_nom == y_rob,
            y_j_jaccard = jaccard(y_nom, y_rob),
            y_j_intersection_size = length(intersect(y_nom, y_rob)),
            y_j_union_size = length(union(y_nom, y_rob)),
            z_js_exact_same = z_nom == z_rob,
            z_js_jaccard = jaccard(z_nom, z_rob),
            z_js_intersection_size = length(intersect(z_nom, z_rob)),
            z_js_union_size = length(union(z_nom, z_rob)),
            nominal_run_dir = nominal["run_dir"],
            robust_run_dir = robust["run_dir"],
        ))
    end
    return DataFrame(rows)
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/compare_station_selections.jl <experiment_dir>")
    exp_dir = normpath(ARGS[1])
    df = build_rows(exp_dir)
    CSV.write(stdout, df)
end

main()
