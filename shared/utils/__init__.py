# NOTE: fm_solvers and fm_solvers_unipc are intentionally NOT imported here
# to avoid pulling in diffusers at module load time (~26s import).
# Import directly from the submodules when needed:
#   from shared.utils.fm_solvers import FlowDPMSolverMultistepScheduler, ...
#   from shared.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
