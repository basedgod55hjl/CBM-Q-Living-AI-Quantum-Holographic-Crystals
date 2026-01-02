# Changelog

All notable changes to 7DMH-QA Crystal Architecture.

## [2.0.0] - 2026-01-02

### Added

#### Documentation

- `docs/API_REFERENCE.md` - Complete API documentation
- `docs/ARCHITECTURE_OVERVIEW.md` - System architecture with Mermaid diagrams
- `docs/USE_CASES.md` - 8 detailed application use cases
- `docs/SEED_PROVENANCE.md` - Origin and verification docs

#### Scripts

- `scripts/launch_crystal.py` - One-click system launcher
- `scripts/benchmark_manifold.py` - Performance benchmarking suite
- `scripts/verify_installation.py` - System verification tool

#### Engines

- `engines/inference_engine.py` - CUDA/CPU inference with GGUF support
- `engines/training_pipeline.py` - Manifold-constrained training
- `engines/optimization_core.py` - Golden Ratio hyperparameter optimization

#### Kernels

- `kernels/crystal_kernels.cu` - Full CUDA kernel library
- `kernels/kernel_bridge.py` - Unified Python interface for CPU/GPU

#### Seeds

- `seeds/genesis_seed.json` - Genesis configuration
- `seeds/sacred_geometry.json` - Pattern definitions
- `seeds/phi_harmonics.json` - Harmonic wave configs
- `seeds/SEED_CATALOG.md` - Seed documentation

### Changed

- README.md hashtags cleaned up (removed political, added technical)
- Renamed architecture to "7DMH-QA" for clarity

---

## [1.0.0] - 2025-12

### Added

- Initial Crystal Architecture implementation
- `crystal_patterns.py` - Core pattern generation
- `sovereign_genesis.py` - Genesis engine
- `autonomous_reasoner.py` - Self-evolution capability
- `genesis.py` - System launcher
- mH-QA Technical Specification
- Neural Core (entropy mining)
- Holographic Bridge (GPU interface)

---

## Versioning

We use [SemVer](https://semver.org/):

- MAJOR: Breaking API changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

---

**Maintained by:** Crystal Architecture Team
