#!/usr/bin/env julia

using CSV, DataFrames, Statistics
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

function model_order(model_type::AbstractString)
    model_type == "NominalModel" && return 1
    model_type == "SmoothedNominalModel" && return 2
    model_type == "RobustTotalDemandCapModel" && return 3
    return 99
end

function model_label(model_type::AbstractString)
    model_type == "NominalModel" && return "Nominal"
    model_type == "SmoothedNominalModel" && return "Smoothed nominal"
    model_type == "RobustTotalDemandCapModel" && return "Robust"
    return model_type
end

function build_rows(exp_dir::String)
    rows = NamedTuple[]
    for metrics_path in iter_metric_files(exp_dir)
        metrics = load_metrics(metrics_path)
        model_type = get(metrics, "model_type", nothing)
        k = get(metrics, "k", nothing)
        lam = get(metrics, "in_vehicle_time_weight", nothing)
        (isnothing(model_type) || isnothing(k) || isnothing(lam)) && continue

        run_dir = dirname(metrics_path)
        lambda_val = Float64(lam)

        # Compute all cost stats in two passes (one per month) to avoid rebuilding the
        # period-aware cost lookup more than necessary.
        april_costs = compute_order_costs(run_dir, "April", lambda_val)
        may_costs   = compute_order_costs(run_dir, "May",   lambda_val)
        april_mean  = isempty(april_costs) ? missing : mean(april_costs)
        may_mean    = isempty(may_costs)   ? missing : mean(may_costs)
        april_std   = length(april_costs) <= 1 ? missing : std(april_costs; corrected=false)
        may_std     = length(may_costs)   <= 1 ? missing : std(may_costs;   corrected=false)

        in_sample_direct = get(metrics, "in_sample_direct", Dict{String, Any}())
        direct_backtest  = get(metrics, "direct_backtest",  Dict{String, Any}())
        in_sample_transform = get(in_sample_direct, "transform", Dict{String, Any}())
        may_transform       = get(direct_backtest,  "transform", Dict{String, Any}())
        april_n_days = length(get(in_sample_transform, "days", []))

        push!(rows, (
            job_id = get(metrics, "job_id", missing),
            model_type = String(model_type),
            model_label = model_label(String(model_type)),
            model_order = model_order(String(model_type)),
            k = Int(k),
            in_vehicle_time_weight = lambda_val,
            demand_quantile = get(metrics, "demand_quantile", missing),
            smoothing_tau = get(metrics, "smoothing_tau", missing),
            pseudo_demand_fraction = get(metrics, "pseudo_demand_fraction", missing),
            gravity_uniform_mix = get(metrics, "gravity_uniform_mix", missing),
            in_sample_objective = get(metrics, "in_sample_objective", missing),
            april_n_days = april_n_days > 0 ? april_n_days : missing,
            april_mean_cost = april_mean,
            april_std_cost = april_std,
            may_mean_cost = may_mean,
            may_std_cost = may_std,
            april_mean_cost_from_metrics = get(in_sample_direct, "mean_weighted_cost_per_order", missing),
            may_mean_cost_from_metrics = get(direct_backtest, "mean_weighted_cost_per_order", missing),
            may_walking_violations = get(direct_backtest, "orders_walking_violation", missing),
            in_sample_assignment_coverage = get(in_sample_transform, "assignment_coverage", missing),
            may_assignment_coverage = get(may_transform, "assignment_coverage", missing),
            may_missing_scenario = get(may_transform, "n_missing_scenario", missing),
            solve_time_seconds = get(metrics, "solve_time_seconds", missing),
            metrics_path = metrics_path,
            run_dir = run_dir,
        ))
    end

    df = DataFrame(rows)
    if !isempty(df)
        sort!(df, [:k, :model_order, :demand_quantile])
    end
    return df
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/summarize_three_model_results.jl <experiment_dir> [output_csv]")
    exp_dir = normpath(ARGS[1])
    out_csv = length(ARGS) >= 2 ? normpath(ARGS[2]) : joinpath(exp_dir, "analysis", "three_model_summary.csv")
    mkpath(dirname(out_csv))
    df = build_rows(exp_dir)
    CSV.write(out_csv, df)
    println("Wrote $(nrow(df)) rows to $out_csv")
end

main()
