#!/usr/bin/env julia

using CSV, DataFrames, Plots
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

ecdf_vals(vals) = isempty(vals) ? (Float64[], Float64[]) : (sort(vals), collect(1:length(vals)) ./ length(vals))

function plot_pair(nominal, robust, out_dir)
    lam = Float64(nominal["in_vehicle_time_weight"])
    napr, nmay = compute_order_costs(String(nominal["run_dir"]), "April", lam), compute_order_costs(String(nominal["run_dir"]), "May", lam)
    rapr, rmay = compute_order_costs(String(robust["run_dir"]), "April", lam), compute_order_costs(String(robust["run_dir"]), "May", lam)
    plt = plot(layout=(1,2), size=(1200,450))
    for (idx, (label, a, b)) in enumerate((("April", napr, rapr), ("May", nmay, rmay)))
        x,y = ecdf_vals(a); !isempty(x) && plot!(plt[idx], x, y; color=:blue, linewidth=2, label="Nominal (n=$(length(a)))")
        x,y = ecdf_vals(b); !isempty(x) && plot!(plt[idx], x, y; color=:orange, linewidth=2, label="Robust (n=$(length(b)))")
        plot!(plt[idx]; title=label, xlabel="Order weighted cost", ylabel="Empirical CDF", gridalpha=0.25)
    end
    plot!(plt; plot_title="k=$(nominal["k"]) | lambda=$(nominal["in_vehicle_time_weight"]) | nominal job $(nominal["job_id"]) vs robust job $(robust["job_id"]) | q=$(robust["comparison_quantile"])")
    out_path = joinpath(out_dir, safe_slug("k$(nominal["k"])_lambda$(nominal["in_vehicle_time_weight"])_nom$(nominal["job_id"])_rob$(robust["job_id"])_q$(robust["comparison_quantile"])") * ".png")
    savefig(plt, out_path)
    return (k=nominal["k"], in_vehicle_time_weight=nominal["in_vehicle_time_weight"], quantile=robust["comparison_quantile"], nominal_job_id=nominal["job_id"], robust_job_id=robust["job_id"], figure_path=out_path, april_nominal_orders=length(napr), april_robust_orders=length(rapr), may_nominal_orders=length(nmay), may_robust_orders=length(rmay))
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/plot_order_cost_cdfs.jl <experiment_dir>")
    exp_dir = normpath(ARGS[1]); out_dir = joinpath(exp_dir, "figures", "order_cost_cdfs"); mkpath(out_dir)
    rows = [plot_pair(n, r, out_dir) for (n, r) in pair_runs(exp_dir)]
    CSV.write(joinpath(out_dir, "manifest.csv"), DataFrame(rows))
    println("Wrote $(length(rows)) CDF figures to $out_dir")
end
main()
