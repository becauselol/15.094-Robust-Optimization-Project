#!/usr/bin/env julia

using Pkg
PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(PROJECT_ROOT, "StationSelection.jl"))
push!(LOAD_PATH, joinpath(PROJECT_ROOT, "MicroTransitSimulator.jl"))

using JSON
using StationSelection
using MicroTransitSimulator

include(joinpath(@__DIR__, "backtest_pipeline_lib.jl"))

if length(ARGS) < 1
    error("Usage: julia scripts/06_aggregate_backtest.jl <backtest_dir>")
end

summary = aggregate_simulation_outputs(abspath(ARGS[1]))
println(JSON.json(summary))
