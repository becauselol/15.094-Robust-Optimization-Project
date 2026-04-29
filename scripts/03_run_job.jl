#!/usr/bin/env julia
"""
Run a single job from a parameter sweep.

Called by 02_submit.sh for SLURM array jobs, or directly for local testing.

Usage:
    julia scripts/03_run_job.jl experiments/<experiment_name> <job_id>

Supported model types (set via cfg["model"]["type"]):
    "NominalModel"              — uses mean demand (E[q_ods]), ClusteringTwoStageODModel
    "RobustTotalDemandCapModel" — total-demand-cap robust counterpart
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "StationSelection.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "MicroTransitSimulator.jl"))
using StationSelection
using MicroTransitSimulator
using CSV, DataFrames, Gurobi, Dates, JSON, TOML, Logging, SHA

include(joinpath(@__DIR__, "backtest_pipeline_lib.jl"))

if length(ARGS) < 2
    error("Usage: julia scripts/03_run_job.jl experiments/<experiment_name> <job_id>")
end

EXP_DIR      = abspath(ARGS[1])
job_id       = parse(Int, ARGS[2])
PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))

println("="^80)
println("Single Job Execution")
println("="^80)
println("Experiment: $EXP_DIR")
println("Job ID:     $job_id")
println()

# ── Load config ───────────────────────────────────────────────────────────────
config_file = joinpath(EXP_DIR, "config", "job_$(job_id).toml")
isfile(config_file) || error("Config not found: $config_file\nRun 01_setup.jl first.")
cfg = TOML.parsefile(config_file)
println("✓ Loaded config: $config_file")

params     = get(cfg, "parameters", Dict())
model_type = get(get(cfg, "model", Dict()), "type", "NominalModel")

println("\nParameters:")
for (k, v) in params; println("  $k = $v"); end
println("  model.type = $model_type")
println()

# ── Run directory ─────────────────────────────────────────────────────────────
timestamp  = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
hash_str   = bytes2hex(sha1(Vector{UInt8}(JSON.json(cfg))))[1:6]

runs_dir = joinpath(EXP_DIR, "runs")
run_dir  = joinpath(runs_dir, "$(timestamp)_$(model_type)_$(hash_str)")
mkpath(run_dir)
cp(config_file, joinpath(run_dir, "config.toml"))

logfile = open(joinpath(run_dir, "logs.txt"), "w")
global_logger(SimpleLogger(logfile))
@info "Run: $run_dir  Job: $job_id  Model: $model_type"

try
    # ── Load spatial data ─────────────────────────────────────────────────────
    @info "Loading data..."
    data_cfg = cfg["data"]
    stations = read_candidate_stations(joinpath(PROJECT_ROOT, data_cfg["station_file"]))
    requests = read_customer_requests(joinpath(PROJECT_ROOT, data_cfg["order_file"]))

    walking_costs = compute_station_pairwise_costs(stations)
    routing_costs = read_routing_costs_from_segments(
        joinpath(PROJECT_ROOT, data_cfg["segment_file"]), stations
    )

    # ── Generate scenarios ────────────────────────────────────────────────────
    @info "Generating scenarios..."
    profile_str = get(params, "scenario_profile", "four_period")
    profile_sym = Symbol(profile_str)
    start_date  = Date(params["start_date"])
    end_date    = Date(params["end_date"])

    scenario_ranges = generate_scenarios_by_profile(start_date, end_date; profile=profile_sym)
    @info "Generated $(length(scenario_ranges)) scenario instances"

    data = create_station_selection_data(
        stations, requests, walking_costs;
        routing_costs=routing_costs, scenarios=scenario_ranges
    )

    # ── Build model ───────────────────────────────────────────────────────────
    k                      = params["k"]
    l                      = get(params, "l", 12)
    lambda_val             = get(params, "in_vehicle_time_weight", 1.0)
    max_walking_distance   = get(params, "max_walking_distance", 300.0)

    gurobi_env = Gurobi.Env()
    start_time = now()

    @info "Building model: $model_type ..."
    result = if model_type == "NominalModel"
        # Use the mean-demand two-stage OD model as the nominal baseline
        nominal_model = ClusteringTwoStageODModel(
            k, l;
            in_vehicle_time_weight = lambda_val,
            max_walking_distance   = max_walking_distance,
        )
        run_opt(nominal_model, data; optimizer_env=gurobi_env, silent=true)

    elseif model_type == "RobustTotalDemandCapModel"
        # Load pre-calibrated demand bounds
        bounds_file = joinpath(PROJECT_ROOT, data_cfg["demand_bounds_file"])
        isfile(bounds_file) || error("demand_bounds_file not found: $bounds_file\n" *
                                     "Run: julia scripts/generate_demand_bounds.jl")
        bounds_raw = JSON.parsefile(bounds_file)

        Q_cap_q   = get(params, "Q_cap_quantile",  nothing)
        q_high_q  = get(params, "q_high_quantile", nothing)

        q_low, q_hat, B = _load_demand_bounds(bounds_raw, data, Q_cap_q, q_high_q)

        robust_model = RobustTotalDemandCapModel(
            k, l;
            in_vehicle_time_weight = lambda_val,
            max_walking_distance   = max_walking_distance,
            q_low = q_low,
            q_hat = q_hat,
            B     = B,
        )
        run_opt(robust_model, data; optimizer_env=gurobi_env, silent=true)

    else
        error("Unknown model type: $model_type")
    end

    elapsed = now() - start_time

    simulation_enabled = Bool(get(get(cfg, "simulation", Dict()), "enabled", true))

    # ── Export solution + prepare backtest artifacts ─────────────────────────
    @info "Exporting optimization variables..."
    export_variables(result, run_dir)

    @info "Preparing transformed orders and direct backtest metrics..."
    backtest_artifacts = prepare_backtest_artifacts(PROJECT_ROOT, cfg, run_dir)

    # ── Run one simulation per day (skipped when simulation.enabled = false) ─
    if simulation_enabled
        simulation_manifest_file = backtest_artifacts["simulation_manifest_file"]
        simulation_manifest = CSV.read(simulation_manifest_file, DataFrame)
        @info "Running $(nrow(simulation_manifest)) daily simulation jobs"
        for row in eachrow(simulation_manifest)
            @info "Simulation day" day_index=row.day_index date=row.date orders=row.order_count
            run_simulation_day_from_manifest(simulation_manifest_file, Int(row.day_index))
        end
        @info "Aggregating simulation outputs..."
        backtest_summary = aggregate_simulation_outputs(backtest_artifacts["backtest_dir"])
    else
        @info "Simulation disabled — skipping daily simulation runs"
        backtest_summary = Dict{String,Any}(
            "manifest_days"   => backtest_artifacts["n_days"],
            "completed_days"  => 0,
            "missing_days"    => String[],
            "passengers"      => Dict{String,Any}(),
            "vehicles"        => Dict{String,Any}(),
            "simulation"      => Dict{String,Any}(),
        )
        if isfile(backtest_artifacts["direct_metrics_file"])
            backtest_summary["direct_backtest"] = JSON.parsefile(backtest_artifacts["direct_metrics_file"])
        end
    end

    # ── Save metrics ──────────────────────────────────────────────────────────
    @info "Saving metrics..."
    metrics = Dict{String,Any}(
        "job_id"              => job_id,
        "model_type"          => model_type,
        "status"              => string(result.termination_status),
        "k"                   => k,
        "l"                   => l,
        "lambda"              => lambda_val,
        "solve_time_seconds"  => result.runtime_sec,
        "timestamp"           => string(timestamp),
        "config_hash"         => hash_str,
        "in_sample_objective" => result.objective_value,
        "simulation_enabled"  => simulation_enabled,
        "backtest_dir"        => backtest_artifacts["backtest_dir"],
        "backtest_manifest_days" => backtest_artifacts["n_days"],
        "direct_backtest_metrics_file" => backtest_artifacts["direct_metrics_file"],
    )
    for (pk, pv) in params
        haskey(metrics, pk) || (metrics[pk] = pv)
    end
    metrics["direct_backtest"] = get(backtest_summary, "direct_backtest", Dict{String, Any}())
    if simulation_enabled
        metrics["simulation_backtest"] = Dict(
            "manifest_days" => backtest_summary["manifest_days"],
            "completed_days" => backtest_summary["completed_days"],
            "missing_days" => backtest_summary["missing_days"],
            "passengers" => backtest_summary["passengers"],
            "vehicles" => backtest_summary["vehicles"],
            "simulation" => backtest_summary["simulation"],
        )
    end

    open(joinpath(run_dir, "metrics.json"), "w") do f
        JSON.print(f, metrics, 4)
    end

    @info "Completed | status=$(result.termination_status) | runtime=$elapsed"
    println("\n✓ Job $job_id completed ($model_type) — $(run_dir)")
    exit(0)

catch e
    @error "Job failed" exception=e
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt); println()
    end
    open(joinpath(run_dir, "error.txt"), "w") do f
        println(f, "Error: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(f, exc, bt); println(f)
        end
    end
    println("\n✗ Job $job_id failed — see $(run_dir)/error.txt")
    exit(1)

finally
    close(logfile)
end

# =============================================================================
# Helpers
# =============================================================================

"""
Parse demand bounds from the JSON file produced by generate_demand_bounds.jl.

JSON keys are strings; convert back to Int (scenario) and Tuple (OD pair).
If Q_cap_quantile or q_high_quantile differ from the stored calibration,
recompute the affected quantities from live scenario data.
Both parameters are independent: either, both, or neither may be overridden.
"""
function _load_demand_bounds(
        bounds_raw::Dict,
        data::StationSelectionData,
        Q_cap_q::Union{Float64, Nothing},
        q_high_q::Union{Float64, Nothing},
    )

    function _parse_od(s::String)
        parts = split(s, ",")
        (parse(Int, parts[1]), parse(Int, parts[2]))
    end

    cal = get(bounds_raw, "calibration", Dict())
    stored_Q_cap_q  = get(cal, "Q_cap_quantile",  nothing)
    stored_q_high_q = get(cal, "q_high_quantile", nothing)

    need_q_hat_recal = !isnothing(q_high_q) && q_high_q != stored_q_high_q
    need_B_recal     = (!isnothing(Q_cap_q) && Q_cap_q != stored_Q_cap_q) || need_q_hat_recal

    if need_q_hat_recal || need_B_recal
        groups = group_scenarios_by_period(data.scenarios)
        effective_q_high = something(q_high_q, get(cal, "q_high_quantile", 0.90))
        effective_Q_cap  = something(Q_cap_q,  get(cal, "Q_cap_quantile",  0.90))
        _, q_hat_new, B_new, _ = compute_demand_bounds(
            groups;
            q_high_quantile = Float64(effective_q_high),
            Q_cap_quantile  = Float64(effective_Q_cap),
        )
        # q_low is always zero; only return the recomputed quantities
        q_low_zero = Dict(s => Dict{Tuple{Int,Int},Float64}(od => 0.0 for od in keys(q_hat_new[s]))
                          for s in keys(q_hat_new))
        return q_low_zero, q_hat_new, B_new
    end

    # Use stored values as-is
    q_low = Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
    q_hat = Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
    for (s_str, od_dict) in bounds_raw["q_low"]
        s = parse(Int, s_str)
        q_low[s] = Dict(_parse_od(k) => Float64(v) for (k, v) in od_dict)
    end
    for (s_str, od_dict) in bounds_raw["q_hat"]
        s = parse(Int, s_str)
        q_hat[s] = Dict(_parse_od(k) => Float64(v) for (k, v) in od_dict)
    end

    # Recalibrate B only if Q_cap_quantile changed (q_hat unchanged)
    if !isnothing(Q_cap_q) && Q_cap_q != stored_Q_cap_q
        groups = group_scenarios_by_period(data.scenarios)
        _, _, B_new, _ = compute_demand_bounds(groups; Q_cap_quantile=Float64(Q_cap_q))
        return q_low, q_hat, B_new
    end

    B_stored = Float64.(bounds_raw["B"])
    return q_low, q_hat, B_stored
end
