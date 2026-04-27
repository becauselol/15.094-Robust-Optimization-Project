#!/usr/bin/env julia
"""
Setup a parameter sweep for one experiment.

Reads base.toml and sweep.toml from the experiment directory,
generates all parameter combinations, writes individual job config
files, and saves a job list for SLURM array submission.

Usage (from project root):
    julia scripts/01_setup.jl experiments/<experiment_name>
"""

using TOML, Dates, CSV, DataFrames, JSON

if length(ARGS) < 1
    error("Usage: julia scripts/01_setup.jl experiments/<experiment_name>")
end

EXP_DIR = abspath(ARGS[1])
PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))

println("="^80)
println("Parameter Sweep Setup")
println("="^80)
println("Experiment directory: $EXP_DIR")
println("Project root:         $PROJECT_ROOT")
println()

# Load configurations
base_path  = joinpath(EXP_DIR, "base.toml")
sweep_path = joinpath(EXP_DIR, "sweep.toml")

isfile(base_path)  || error("base.toml not found at: $base_path")
isfile(sweep_path) || error("sweep.toml not found at: $sweep_path")

base  = TOML.parsefile(base_path)
sweep = TOML.parsefile(sweep_path)

println("Loaded:")
println("  base:  $base_path")
println("  sweep: $sweep_path")
println()

# Generate Cartesian product of sweep parameters
function generate_combinations(d::Dict)
    keys_vec = collect(keys(d))
    vals_vec  = collect(values(d))
    combos    = vec(collect(Iterators.product(vals_vec...)))
    return [Dict(zip(keys_vec, c)) for c in combos]
end

param_combos = generate_combinations(sweep)
println("Generated $(length(param_combos)) parameter combinations")
println()

# Create config directory and write per-job TOML files
config_dir = joinpath(EXP_DIR, "config")
mkpath(config_dir)

for (i, params) in enumerate(param_combos)
    job_cfg = deepcopy(base)
    # Sweep params override or extend the [parameters] table in base
    params_table = get(job_cfg, "parameters", Dict())
    merge!(params_table, params)
    job_cfg["parameters"] = params_table

    config_file = joinpath(config_dir, "job_$(i).toml")
    open(config_file, "w") do f
        TOML.print(f, job_cfg)
    end
end
println("✓ Saved $(length(param_combos)) config files → config/")

# Save job list CSV
jobs = [merge(Dict("job_id" => i), combo) for (i, combo) in enumerate(param_combos)]
jobs_df = DataFrame(jobs)
select!(jobs_df, :job_id, :)

job_list_path = joinpath(config_dir, "job_list.csv")
CSV.write(job_list_path, jobs_df)
println("✓ Saved job list:      $job_list_path")

# Save plain-text job index for SLURM
jobs_txt_path = joinpath(config_dir, "jobs.txt")
open(jobs_txt_path, "w") do f
    for i in 1:length(param_combos)
        println(f, i)
    end
end
println("✓ Saved SLURM indices: $jobs_txt_path")

# Save setup metadata
metadata = Dict(
    "created"           => string(now()),
    "total_jobs"        => length(jobs),
    "sweep_parameters"  => collect(keys(sweep)),
    "base_config"       => base_path,
    "sweep_config"      => sweep_path,
    "julia_version"     => string(VERSION),
)
open(joinpath(EXP_DIR, "setup_metadata.json"), "w") do f
    JSON.print(f, metadata, 2)
end
println("✓ Saved setup_metadata.json")

println()
println("="^80)
println("Summary: $(length(jobs)) jobs")
println()
println("Sweep parameters:")
for (p, vals) in sweep
    println("  $p: $vals  ($(length(vals)) values)")
end
println()
println("Next steps:")
println("  1. Review:  $job_list_path")
println("  2. Locally: julia scripts/03_run_job.jl $EXP_DIR <job_id>")
println("  3. SLURM:   sbatch experiments/<experiment_name>/02_submit.sh")
println("="^80)
