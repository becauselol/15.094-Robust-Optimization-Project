#!/usr/bin/env julia

using Pkg
PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(PROJECT_ROOT, "StationSelection.jl"))
push!(LOAD_PATH, joinpath(PROJECT_ROOT, "MicroTransitSimulator.jl"))

using JSON
using TOML
using StationSelection
using MicroTransitSimulator

include(joinpath(@__DIR__, "backtest_pipeline_lib.jl"))

if length(ARGS) < 1
    error("Usage: julia scripts/04_prepare_backtest.jl <run_dir>")
end

run_dir = abspath(ARGS[1])
config_file = joinpath(run_dir, "config.toml")
isfile(config_file) || error("Missing config.toml in $run_dir")
cfg = TOML.parsefile(config_file)

artifacts = prepare_backtest_artifacts(PROJECT_ROOT, cfg, run_dir)
println(JSON.json(artifacts))
