#!/usr/bin/env julia
# Evaluates experiment station selections against an external May orders CSV
# (e.g. full-OD zhuzhou_data_40) instead of each run's own backtest orders.
# Outputs per-scenario summary statistics.

using CSV, DataFrames, Statistics, Dates
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

function build_rows(exp_dir::String, may_orders::DataFrame)
    rows = NamedTuple[]
    for metrics_path in iter_metric_files(exp_dir)
        metrics = load_metrics(metrics_path)
        model_type = get(metrics, "model_type", nothing)
        k = get(metrics, "k", nothing)
        lam = get(metrics, "in_vehicle_time_weight", nothing)
        (isnothing(model_type) || isnothing(k) || isnothing(lam)) && continue

        run_dir = dirname(metrics_path)
        lambda_val = Float64(lam)
        mtype = String(model_type)
        quantile = get(metrics, "demand_quantile", missing)

        costs_by_period = compute_order_costs_by_scenario_from_df(run_dir, may_orders, lambda_val)
        all_costs = vcat(values(costs_by_period)...)

        push!(rows, (
            job_id            = get(metrics, "job_id", missing),
            model_type        = mtype,
            model_label       = model_label(mtype),
            model_order       = model_order(mtype),
            k                 = Int(k),
            in_vehicle_time_weight = lambda_val,
            demand_quantile   = quantile,
            scenario          = "all",
            n_orders          = length(all_costs),
            mean_cost         = isempty(all_costs) ? missing : mean(all_costs),
            std_cost          = length(all_costs) <= 1 ? missing : std(all_costs; corrected=false),
            metrics_path      = metrics_path,
        ))

        for period in sort(collect(keys(costs_by_period)))
            costs = costs_by_period[period]
            push!(rows, (
                job_id            = get(metrics, "job_id", missing),
                model_type        = mtype,
                model_label       = model_label(mtype),
                model_order       = model_order(mtype),
                k                 = Int(k),
                in_vehicle_time_weight = lambda_val,
                demand_quantile   = quantile,
                scenario          = period,
                n_orders          = length(costs),
                mean_cost         = isempty(costs) ? missing : mean(costs),
                std_cost          = length(costs) <= 1 ? missing : std(costs; corrected=false),
                metrics_path      = metrics_path,
            ))
        end
    end

    df = DataFrame(rows)
    if !isempty(df)
        sort!(df, [:k, :model_order, :demand_quantile, :scenario])
    end
    return df
end

function main()
    length(ARGS) >= 2 || error("""
    Usage: julia --project=StationSelection.jl scripts/evaluate_full_od_may.jl \\
               <exp_dir> <full_od_orders_csv> [output_csv]
    """)
    exp_dir    = normpath(ARGS[1])
    orders_path = normpath(ARGS[2])
    out_csv    = length(ARGS) >= 3 ? normpath(ARGS[3]) :
                 joinpath(exp_dir, "analysis", "full_od_may_evaluation.csv")
    mkpath(dirname(out_csv))

    println("Loading May orders from $orders_path ...")
    orders_df = CSV.read(orders_path, DataFrame)
    orders_df.order_time = DateTime.(orders_df.order_time, "yyyy-mm-dd HH:MM:SS")
    may_orders = filter(r -> month(r.order_time) == 5, orders_df)
    println("  $(nrow(may_orders)) May orders")

    df = build_rows(exp_dir, may_orders)
    CSV.write(out_csv, df)
    println("Wrote $(nrow(df)) rows to $out_csv")
end

main()
