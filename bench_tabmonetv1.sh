#!/bin/bash

uv run python scripts/run_tabarena_lite.py
tar czf results.tar.gz tabarena_out
