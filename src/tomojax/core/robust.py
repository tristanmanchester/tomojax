"""Robust scale constants shared by every package."""

# A median absolute deviation times this is a standard deviation for Gaussian data:
# 1 / the standard normal's 0.75 quantile.
MAD_TO_SD = 1 / 0.6744897501960817
