# JAX/Optax Compatibility Fix - September 3, 2025

## Problem Summary

When attempting to add `optax` to the TomoJAX project, JAX imports began failing with:
```
ImportError: libgrpc.so.46: cannot open shared object file: No such file or directory
```

This occurred despite the environment working perfectly before adding optax.

## Root Cause Analysis

The issue was a **version compatibility cascade**:

1. **Working State**: JAX 0.4.x was compatible with conda-installed gRPC libraries (version 2301.0.0)
2. **Breaking Change**: Adding `optax = "*"` to pixi.toml triggered an upgrade to JAX 0.7.1
3. **Library Mismatch**: JAX 0.7.1 expected newer Abseil libraries (version 2501.0.0) but conda environment only had 2301.0.0
4. **Import Failure**: Missing shared libraries caused JAX to fail at import time

### Key Discovery
When `pixi.toml` specified `jax = { version = ">=0.4.23", extras = ["cuda12"] }`, pip resolved this to the latest available version (0.7.1) when installing fresh. The pixi.lock file normally prevents this, but adding optax forced a lock file update.

## Investigation Process

### 1. Initial Debugging
- Identified missing `libgrpc.so.46` and `libabsl_*` libraries 
- Found existing libraries with different version numbers (29.0.0 vs 46, 2301.0.0 vs 2501.0.0)
- Attempted symlink workaround (partially successful but brittle)

### 2. Version Analysis  
- Discovered JAX had upgraded from 0.4.x to 0.7.1
- Found that current optax (0.2.5) requires `jax>=0.5.3`
- Realized the problem wasn't the libraries themselves, but version expectations

### 3. Systematic Fix
- Reset pixi.toml and pixi.lock to clean state
- Determined JAX 0.5.3 was the sweet spot: newer than 0.4.x but older than problematic 0.7.x
- Updated version constraints to be compatible

## Solution Implementation

### Final Configuration (pixi.toml)
```toml
[pypi-dependencies]
jax = { version = ">=0.5.3,<0.6", extras = ["cuda12"] }
optax = "*"
```

### Version Resolution
- **JAX**: 0.5.3 (compatible with both conda gRPC libraries and optax requirements)
- **Optax**: 0.2.5 (latest, requires jax>=0.5.3)
- **Result**: No shared library conflicts, full CUDA support maintained

## Key Lessons Learned

1. **Version Constraints Matter**: Open-ended version specs (`>=0.4.23`) can cause problems when new major versions are released
   
2. **Dependency Cascades**: Adding one package (optax) can trigger upgrades of foundational packages (JAX) with unexpected consequences

3. **Lock Files Are Critical**: The pixi.lock file prevents unwanted upgrades, but gets updated when new dependencies are added

4. **Library Version Mismatches**: Modern PyPI packages may expect newer system libraries than provided by conda/system package managers

5. **Sweet Spot Versions**: Sometimes the solution isn't the newest or oldest version, but a middle version that satisfies all constraints

## Prevention Strategies

1. **Pin Major Versions**: Use constraints like `<0.6` to prevent unexpected major version jumps
2. **Test After Adding Dependencies**: Always verify existing functionality after adding new packages  
3. **Understand Dependency Trees**: Check what versions new packages require before adding them
4. **Environment Snapshots**: Keep working pixi.lock files backed up

## Performance Validation

After the fix, we validated the environment with a gradient descent vs L-BFGS comparison:

```
Problem: 10,000-dimensional ill-conditioned quadratic (κ=10⁴)
Results:
- GD (conservative): 100,000 iterations, no convergence  
- GD (aggressive): 100,000 iterations, no convergence
- L-BFGS: 1,291 iterations, converged in 2.72s

Conclusion: L-BFGS is ~77x more efficient for this class of problems
```

## Files Modified

- `pixi.toml`: Updated JAX version constraint and added optax
- `alignment-testing/gd_vs_l-bfgs.py`: Fixed block_until_ready() tuple issue and improved comparison

## Environment Status

✅ **WORKING**: JAX 0.5.3 + optax 0.2.5 + CUDA support  
✅ **Verified**: Full GPU functionality with proper CUDA module loading  
✅ **Performance**: Comparable to original working state  

---
*This fix was implemented on September 3, 2025, resolving compatibility issues introduced when adding optax optimization library to the TomoJAX project.*