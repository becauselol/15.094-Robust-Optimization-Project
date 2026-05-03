#!/usr/bin/env julia

using CSV, DataFrames, Plots
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

ecdf_vals(vals) = isempty(vals) ? (Float64[], Float64[]) : (sort(vals), collect(1:length(vals)) ./ length(vals))

function plot_pair_month(nominal, robust, month, out_dir)
    lam = Float64(nominal["in_vehicle_time_weight"])
    nom = compute_order_costs_by_scenario(String(nominal["run_dir"]), month, lam)
    rob = compute_order_costs_by_scenario(String(robust["run_dir"]), month, lam)
    plt = plot(layout=(2,2), size=(1200,800))
    for (idx, label) in enumerate(("period_1", "period_2", "period_3", "period_4"))
        x,y = ecdf_vals(get(nom, label, Float64[])); !isempty(x) && plot!(plt[idx], x, y; color=:blue, linewidth=2, label="Nominal (n=$(length(x)))")
        x,y = ecdf_vals(get(rob, label, Float64[])); !isempty(x) && plot!(plt[idx], x, y; color=:orange, linewidth=2, label="Robust (n=$(length(x)))")
        plot!(plt[idx]; title=label, xlabel="Order weighted cost", ylabel="Empirical CDF", gridalpha=0.25)
    end
    plot!(plt; plot_title="$(month) order-cost CDFs by scenario | k=$(nominal["k"]) | lambda=$(nominal["in_vehicle_time_weight"]) | nominal job $(nominal["job_id"]) vs robust job $(robust["job_id"]) | q=$(robust["comparison_quantile"])")
    out_path = joinpath(out_dir, safe_slug("$(month)_k$(nominal["k"])_lambda$(nominal["in_vehicle_time_weight"])_nom$(nominal["job_id"])_rob$(robust["job_id"])_q$(robust["comparison_quantile"])") * ".png")
    savefig(plt, out_path)
    return (month=month, k=nominal["k"], in_vehicle_time_weight=nominal["in_vehicle_time_weight"], quantile=robust["comparison_quantile"], nominal_job_id=nominal["job_id"], robust_job_id=robust["job_id"], figure_path=out_path)
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/plot_order_cost_cdfs_by_scenario.jl <experiment_dir>")
    exp_dir = normpath(ARGS[1]); out_dir = joinpath(exp_dir, "figures", "order_cost_cdfs_by_scenario"); mkpath(out_dir)
    rows = NamedTuple[]
    for (n, r) in pair_runs(exp_dir), month in ("April", "May")
        push!(rows, plot_pair_month(n, r, month, out_dir))
    end
    CSV.write(joinpath(out_dir, "manifest.csv"), DataFrame(rows))
    println("Wrote $(length(rows)) order-cost CDF-by-scenario figures to $out_dir")
end
main()
