"""
ChipSAT: Macro Placement with CP-SAT and ML-Guided Search.

Core modules:
  - cpsat_solver: OR-Tools CP-SAT wrapper (legalize, solve_subset, HPWL)
  - lns_solver: Large Neighborhood Search orchestrator
  - benchmark_loader: ICCAD04 BookShelf parser
  - gnn_layers: GNN building blocks (message passing, MLP, scatter)
  - net_spatial_gnn: Dual-stream topology+spatial GNN
  - greedy_placer: Sequential heatmap-based placement decoder
"""
