#!/usr/bin/env julia

using CSV, DataFrames, Plots
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

function plot_run(metrics_path::String, out_dir::String; bins::Int=60)
    m = load_metrics(metrics_path)
    run_dir = dirname(metrics_path)
    lam = Float64(get(m, "in_vehicle_time_weight", 0.0))
    april = compute_order_costs(run_dir, "April", lam)
    may = compute_order_costs(run_dir, "May", lam)
    plt = plot(layout=(1,2), size=(1200,450))
    histogram!(plt[1], april; bins=bins, color=:blue, alpha=0.8, label="", xlabel="Order weighted cost", ylabel="Count", title="April | n=$(length(april))")
    histogram!(plt[2], may; bins=bins, color=:orange, alpha=0.8, label="", xlabel="Order weighted cost", ylabel="Count", title="May | n=$(length(may))")
    plot!(plt; plot_title="job $(get(m, "job_id", "unknown")) | $(get(m, "model_type", "unknown")) | k=$(get(m, "k", "na")) | lambda=$(get(m, "in_vehicle_time_weight", "na"))")
    out_path = joinpath(out_dir, safe_slug("job$(get(m, "job_id", "unknown"))_$(get(m, "model_type", "unknown"))_k$(get(m, "k", "na"))_lambda$(get(m, "in_vehicle_time_weight", "na"))_q$(get(m, "demand_quantile", get(m, "Q_cap_quantile", "na")))_$(basename(run_dir))") * ".png")
    savefig(plt, out_path)
    return (job_id=get(m, "job_id", missing), model_type=get(m, "model_type", missing), k=get(m, "k", missing), in_vehicle_time_weight=get(m, "in_vehicle_time_weight", missing), demand_quantile=get(m, "demand_quantile", get(m, "Q_cap_quantile", missing)), run_dir=run_dir, figure_path=out_path, april_orders=length(april), may_orders=length(may))
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/plot_order_cost_histograms.jl <experiment_dir>")
    exp_dir = normpath(ARGS[1]); out_dir = joinpath(exp_dir, "figures", "order_cost_histograms"); mkpath(out_dir)
    rows = [plot_run(mp, out_dir) for mp in iter_metric_files(exp_dir)]
    CSV.write(joinpath(out_dir, "manifest.csv"), DataFrame(rows))
    println("Wrote $(length(rows)) histogram figures to $out_dir")
end
main()
