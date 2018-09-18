#!/usr/bin/env bash

set -e

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

ROOT="/home/kgt/workspace/train_data"
echo "Saving data in: ""$ROOT"

echo "Processing: ""$ROOT"/"$1"
cat "$ROOT"/"$1" | tr _ ' ' | tr - ' ' | tr – ' ' |  sed 's/[][\/$*.^|@#{}~&()_:°;%+"='\'',`><?!-]/ /g' | tr -s ' '  > "${ROOT}"/"$2"

