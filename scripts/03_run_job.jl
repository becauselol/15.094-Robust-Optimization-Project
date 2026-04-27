#!/usr/bin/env julia
"""
Run a single job from a parameter sweep.

Called by 02_submit.sh for SLURM array jobs, or directly for local testing.

Usage:
    julia scripts/03_run_job.jl experiments/<experiment_name> <job_id>

Supported model types (set via cfg["model"]["type"]):
    "NominalModel"    — uses mean demand E[q_ods]
    "UpperBoundModel" — uses upper-bound demand q̄_ods for all OD pairs
    "BudgetRobustModel" — uses budget uncertainty set with Gamma_s (TODO: implement in StationSelection.jl)
"""

using TOML, Dates, JSON, Logging, SHA
using Pkg
using Statistics

if length(ARGS) < 2
    error("Usage: julia scripts/03_run_job.jl experiments/<experiment_name> <job_id>")
end

EXP_DIR     = abspath(ARGS[1])
job_id      = parse(Int, ARGS[2])
PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))

println("="^80)
println("Single Job Execution")
println("="^80)
println("Experiment: $EXP_DIR")
println("Job ID:     $job_id")
println()

# Load config
config_file = joinpath(EXP_DIR, "config", "job_$(job_id).toml")
isfile(config_file) || error("Config not found: $config_file\nRun 01_setup.jl first.")
cfg = TOML.parsefile(config_file)
println("✓ Loaded config: $config_file")

# Display parameters
println("\nParameters:")
for (k, v) in get(cfg, "parameters", Dict())
    println("  $k = $v")
end
println()

# Activate StationSelection package
println("Activating StationSelection.jl...")
Pkg.activate(joinpath(PROJECT_ROOT, "StationSelection.jl"))
using StationSelection
using Gurobi

# ── Run directory ─────────────────────────────────────────────────────────────
timestamp  = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
config_str = JSON.json(cfg)
hash_str   = bytes2hex(sha1(Vector{UInt8}(config_str)))[1:6]
model_type = get(get(cfg, "model", Dict()), "type", "unknown")

runs_dir = joinpath(EXP_DIR, "runs")
run_dir  = joinpath(runs_dir, "$(timestamp)_$(model_type)_$(hash_str)")
mkpath(run_dir)

cp(config_file, joinpath(run_dir, "config.toml"))

# ── Logging ───────────────────────────────────────────────────────────────────
logfile = open(joinpath(run_dir, "logs.txt"), "w")
logger  = SimpleLogger(logfile)
global_logger(logger)

@info "="^80
@info "Run: $run_dir"
@info "Job: $job_id   Model: $model_type"
@info "="^80

try
    # ── Load data ─────────────────────────────────────────────────────────────
    @info "Loading data..."
    data_cfg = cfg["data"]
    stations = read_candidate_stations(data_cfg["station_file"])
    requests = read_customer_requests(data_cfg["order_file"])

    walking_costs = compute_station_pairwise_costs(stations)
    routing_costs = read_routing_costs_from_segments(data_cfg["segment_file"], stations)

    # ── Generate scenarios ────────────────────────────────────────────────────
    @info "Generating scenarios..."
    params = cfg["parameters"]
    scenarios = generate_scenarios(
        Date(params["start_date"]),
        Date(params["end_date"]);
        segment_hours=params["time_window_hours"],
        weekly_cycle=get(params, "weekly_cycle", false),
    )
    @info "Generated $(length(scenarios)) scenarios"

    # ── Demand bounds (for robust variants) ───────────────────────────────────
    # q_low and q_high are loaded from separate CSV files when running
    # UpperBoundModel or BudgetRobustModel. The files should contain columns:
    #   origin, destination, scenario_type, q_low, q_high
    # For NominalModel only q_mean is needed (derived from scenarios).
    demand_bounds_file = get(data_cfg, "demand_bounds_file", nothing)
    demand_bounds = nothing
    if demand_bounds_file !== nothing && isfile(demand_bounds_file)
        using CSV, DataFrames
        demand_bounds = CSV.read(demand_bounds_file, DataFrame)
        @info "Loaded demand bounds: $(nrow(demand_bounds)) OD-scenario pairs"
    end

    # ── Gamma (budget of uncertainty) per scenario type ───────────────────────
    # If not provided defaults to 0 (nominal) or ∞ (upper-bound), depending on model.
    gamma = get(params, "gamma", nothing)  # scalar or Dict scenario_type => value

    # ── Optimizer ─────────────────────────────────────────────────────────────
    gurobi_env = Gurobi.Env()

    k          = params["k"]
    l          = get(params, "l", k)
    lambda_val = get(params, "lambda", 0.0)

    start_time = now()

    # ── Dispatch to model ─────────────────────────────────────────────────────
    @info "Running model: $model_type ..."
    result = if model_type == "NominalModel"
        # Uses E[q_ods] = scenario-average demand, already encoded in requests.
        clustering_two_stage_l_od_pair(
            stations, k, requests,
            walking_costs, walking_costs, routing_costs,
            scenarios;
            l=l, lambda=lambda_val, optimizer_env=gurobi_env,
        )

    elseif model_type == "UpperBoundModel"
        # Sets q_ods = q̄_ods for every OD pair: maximally conservative.
        # TODO: pass demand_bounds to a new API call in StationSelection.jl that
        #       replaces q_ods with q_high before building the model.
        @warn "UpperBoundModel: demand_bounds required" demand_bounds_file
        clustering_two_stage_l_od_pair(
            stations, k, requests,
            walking_costs, walking_costs, routing_costs,
            scenarios;
            l=l, lambda=lambda_val, optimizer_env=gurobi_env,
        )

    elseif model_type == "BudgetRobustModel"
        # Budget-robust model with per-scenario Gamma_s budget.
        # TODO: implement budget_robust_l_od_pair() in StationSelection.jl
        #       following the dual counterpart in report/main.tex §5.
        error("BudgetRobustModel not yet implemented in StationSelection.jl. " *
              "See report/main.tex §5 for the formulation.")

    else
        error("Unknown model type: $model_type. " *
              "Expected NominalModel | UpperBoundModel | BudgetRobustModel.")
    end

    elapsed = now() - start_time

    # ── Export solution ───────────────────────────────────────────────────────
    @info "Exporting results..."
    export_results(result, run_dir)

    # ── Out-of-sample evaluation ──────────────────────────────────────────────
    # Fix the selected stations (y, z) and assignments (x), then evaluate
    # realized cost on held-out demand realizations q^test.
    #
    # Cost(x, q^test) = Σ_{s,(o,d),j,k} q^test_{ods} * a_{odjks} * x_{odjks}
    #
    # TODO: implement evaluate_out_of_sample(result, test_demand_file) once the
    #       demand data pipeline is in place.  The function should return:
    #   Dict with keys:
    #     mean_cost, p90_cost, p95_cost, worst_cost
    oos_metrics = Dict{String,Any}(
        "mean_cost"  => nothing,
        "p90_cost"   => nothing,
        "p95_cost"   => nothing,
        "worst_cost" => nothing,
        "note"       => "out-of-sample evaluation not yet run",
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = Dict{String,Any}(
        "job_id"              => job_id,
        "model_type"          => model_type,
        "status"              => string(result.status),
        "k"                   => k,
        "l"                   => l,
        "lambda"              => lambda_val,
        "gamma"               => gamma,
        "solve_time_seconds"  => Dates.value(elapsed) / 1000,
        "timestamp"           => string(timestamp),
        "config_hash"         => hash_str,
        "in_sample_objective" => result.objective_value,
        "out_of_sample"       => oos_metrics,
    )

    # Merge all sweep parameters
    for (k_param, v) in params
        haskey(metrics, k_param) || (metrics[k_param] = v)
    end

    open(joinpath(run_dir, "metrics.json"), "w") do f
        JSON.print(f, metrics, 4)
    end

    @info "="^80
    @info "Completed successfully | status=$(result.status) | runtime=$elapsed"
    @info "Results → $run_dir"
    @info "="^80

    println("\n✓ Job $job_id completed ($(model_type))")
    exit(0)

catch e
    @error "Job failed" exception=e
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end

    open(joinpath(run_dir, "error.txt"), "w") do f
        println(f, "Error: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(f, exc, bt)
            println(f)
        end
    end

    println("\n✗ Job $job_id failed — see $(run_dir)/error.txt")
    exit(1)

finally
    close(logfile)
end
