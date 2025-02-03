There's a simple simulator that uses `Agents.jl` directly in `scripts/simpleex1.jl`, this git repo also contains a julia environment which lists dependencies etc.
To run the example first make sure you have julia version above 1.11 by running `julia --version`.
If so then run `julia --project=./` in the git root then run `]instantiate` and perhaps also `]update` just to make sure, these will likely take a while they just install all the dependencies.
To then actually run the code there's many ways, perhaps the simplest is to continue in the repl/shell from before and run `using Revise; includet("scripts/simpleex1.jl")` which will include all the stuff from the file and then `runmmtest()` which start an example simulation run.
