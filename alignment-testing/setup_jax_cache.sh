#!/bin/bash
# Setup script for JAX compilation cache to improve performance across runs

# Create JAX cache directory
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache_tomojax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "JAX compilation cache enabled at: $JAX_COMPILATION_CACHE_DIR"
echo "Add the following to your shell profile (.bashrc/.zshrc) to make permanent:"
echo "export JAX_COMPILATION_CACHE_DIR=\"$JAX_COMPILATION_CACHE_DIR\""

# Optional: Enable compilation logging for debugging
# export JAX_LOG_COMPILES=1

# Run the provided command with JAX cache enabled
if [ "$#" -gt 0 ]; then
    echo "Running command with JAX cache: $@"
    exec "$@"
fi