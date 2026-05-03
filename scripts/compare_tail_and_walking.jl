#!/usr/bin/env julia

using CSV, DataFrames, Statistics
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

qtile(vals, q) = isempty(vals) ? nothing : quantile(vals, q)

function build_rows(exp_dir::String)
    rows = NamedTuple[]
    for (nominal, robust) in pair_runs(exp_dir)
        lam = Float64(nominal["in_vehicle_time_weight"])
        ntheo = theoretical_walking_violation_rates(String(nominal["run_dir"]))
        rtheo = theoretical_walking_violation_rates(String(robust["run_dir"]))
        naps = compute_order_costs(String(nominal["run_dir"]), "April", lam)
        nmys = compute_order_costs(String(nominal["run_dir"]), "May", lam)
        raps = compute_order_costs(String(robust["run_dir"]), "April", lam)
        rmys = compute_order_costs(String(robust["run_dir"]), "May", lam)
        push!(rows, (
            k = nominal["k"],
            in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
            quantile = robust["comparison_quantile"],
            nominal_job_id = nominal["job_id"],
            robust_job_id = robust["job_id"],
            nominal_april_p90 = qtile(naps, 0.9),
            robust_april_p90 = qtile(raps, 0.9),
            robust_over_nominal_april_p90 = isempty(naps) ? nothing : qtile(raps, 0.9) / qtile(naps, 0.9),
            nominal_april_p95 = qtile(naps, 0.95),
            robust_april_p95 = qtile(raps, 0.95),
            robust_over_nominal_april_p95 = isempty(naps) ? nothing : qtile(raps, 0.95) / qtile(naps, 0.95),
            nominal_april_p99 = qtile(naps, 0.99),
            robust_april_p99 = qtile(raps, 0.99),
            robust_over_nominal_april_p99 = isempty(naps) ? nothing : qtile(raps, 0.99) / qtile(naps, 0.99),
            nominal_may_p90 = qtile(nmys, 0.9),
            robust_may_p90 = qtile(rmys, 0.9),
            robust_over_nominal_may_p90 = isempty(nmys) ? nothing : qtile(rmys, 0.9) / qtile(nmys, 0.9),
            nominal_may_p95 = qtile(nmys, 0.95),
            robust_may_p95 = qtile(rmys, 0.95),
            robust_over_nominal_may_p95 = isempty(nmys) ? nothing : qtile(rmys, 0.95) / qtile(nmys, 0.95),
            nominal_may_p99 = qtile(nmys, 0.99),
            robust_may_p99 = qtile(rmys, 0.99),
            robust_over_nominal_may_p99 = isempty(nmys) ? nothing : qtile(rmys, 0.99) / qtile(nmys, 0.99),
            nominal_april_walking_violation_rate = get_realized_walking_violation_rate(String(nominal["run_dir"]), "April"),
            robust_april_walking_violation_rate = get_realized_walking_violation_rate(String(robust["run_dir"]), "April"),
            nominal_may_walking_violation_rate = get_realized_walking_violation_rate(String(nominal["run_dir"]), "May"),
            robust_may_walking_violation_rate = get_realized_walking_violation_rate(String(robust["run_dir"]), "May"),
            nominal_theoretical_walking_violation_rate_period_1 = get(ntheo, "period_1", missing),
            robust_theoretical_walking_violation_rate_period_1 = get(rtheo, "period_1", missing),
            nominal_theoretical_walking_violation_rate_period_2 = get(ntheo, "period_2", missing),
            robust_theoretical_walking_violation_rate_period_2 = get(rtheo, "period_2", missing),
            nominal_theoretical_walking_violation_rate_period_3 = get(ntheo, "period_3", missing),
            robust_theoretical_walking_violation_rate_period_3 = get(rtheo, "period_3", missing),
            nominal_theoretical_walking_violation_rate_period_4 = get(ntheo, "period_4", missing),
            robust_theoretical_walking_violation_rate_period_4 = get(rtheo, "period_4", missing),
            nominal_theoretical_walking_violation_rate_scenario_avg = get(ntheo, "scenario_avg", missing),
            robust_theoretical_walking_violation_rate_scenario_avg = get(rtheo, "scenario_avg", missing),
            nominal_metrics_path = nominal["metrics_path"],
            robust_metrics_path = robust["metrics_path"],
        ))
    end
    return DataFrame(rows)
end

main() = (length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/compare_tail_and_walking.jl <experiment_dir>"); CSV.write(stdout, build_rows(normpath(ARGS[1]))))
main()
