#!/usr/bin/env julia

using CSV, DataFrames, Statistics
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

function build_rows(exp_dir::String)
    rows = NamedTuple[]
    for (nominal, robust) in pair_runs(exp_dir)
        for month in ("April", "May")
            lam = Float64(nominal["in_vehicle_time_weight"])
            n_total = get_order_cost_total(String(nominal["run_dir"]), month, lam)
            r_total = get_order_cost_total(String(robust["run_dir"]), month, lam)
            if isnothing(n_total) || isnothing(r_total)
                continue
            end
            n_mean = get_order_cost_mean(String(nominal["run_dir"]), month, lam)
            r_mean = get_order_cost_mean(String(robust["run_dir"]), month, lam)
            n_std = get_order_cost_std(String(nominal["run_dir"]), month, lam)
            r_std = get_order_cost_std(String(robust["run_dir"]), month, lam)
            n_theo = get_theoretical_od_metrics(String(nominal["run_dir"]))
            r_theo = get_theoretical_od_metrics(String(robust["run_dir"]))
            push!(rows, (
                k = nominal["k"],
                in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
                month = month,
                quantile = robust["comparison_quantile"],
                nominal_job_id = nominal["job_id"],
                robust_job_id = robust["job_id"],
                nominal_mean_cost_per_order = n_mean,
                robust_mean_cost_per_order = r_mean,
                robust_over_nominal_mean_cost_per_order = isnothing(n_mean) || n_mean == 0 ? nothing : r_mean / n_mean,
                nominal_order_cost_std = n_std,
                robust_order_cost_std = r_std,
                robust_over_nominal_order_cost_std = isnothing(n_std) || n_std == 0 ? nothing : r_std / n_std,
                nominal_theoretical_mean_od_mean_cost_unweighted = n_theo.mean_od_mean_cost_theoretical,
                robust_theoretical_mean_od_mean_cost_unweighted = r_theo.mean_od_mean_cost_theoretical,
                robust_over_nominal_theoretical_mean_od_mean_cost_unweighted = isnothing(n_theo.mean_od_mean_cost_theoretical) || n_theo.mean_od_mean_cost_theoretical == 0 ? nothing : r_theo.mean_od_mean_cost_theoretical / n_theo.mean_od_mean_cost_theoretical,
                nominal_theoretical_std_od_mean_cost_unweighted = n_theo.std_od_mean_cost_theoretical,
                robust_theoretical_std_od_mean_cost_unweighted = r_theo.std_od_mean_cost_theoretical,
                robust_over_nominal_theoretical_std_od_mean_cost_unweighted = isnothing(n_theo.std_od_mean_cost_theoretical) || n_theo.std_od_mean_cost_theoretical == 0 ? nothing : r_theo.std_od_mean_cost_theoretical / n_theo.std_od_mean_cost_theoretical,
                nominal_metrics_path = nominal["metrics_path"],
                robust_metrics_path = robust["metrics_path"],
            ))
        end
    end
    return DataFrame(rows)
end

main() = (length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/compare_weighted_total_cost.jl <experiment_dir>"); CSV.write(stdout, build_rows(normpath(ARGS[1]))))
main()
