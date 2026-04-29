#!/usr/bin/env julia

using Pkg
PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(PROJECT_ROOT, "StationSelection.jl"))
push!(LOAD_PATH, joinpath(PROJECT_ROOT, "MicroTransitSimulator.jl"))

using StationSelection
using MicroTransitSimulator

include(joinpath(@__DIR__, "backtest_pipeline_lib.jl"))

if length(ARGS) < 2
    error("Usage: julia scripts/05_run_simulation_day.jl <manifest_file> <day_index>")
end

manifest_file = abspath(ARGS[1])
day_index = parse(Int, ARGS[2])

run_simulation_day_from_manifest(manifest_file, day_index)
