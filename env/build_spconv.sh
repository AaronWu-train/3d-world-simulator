#!/usr/bin/env bash
set -euo pipefail
export CUMM_CUDA_VERSION=${CUMM_CUDA_VERSION:-12.8} CUMM_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0} CUMM_DISABLE_JIT=1 SPCONV_DISABLE_JIT=1
pip uninstall -y spconv-cu126 spconv cumm cumm-cu126 2>/dev/null || true
pip install pccm ccimport
W=$(mktemp -d); cd "$W"
git clone --depth 1 --branch v0.7.11 https://github.com/FindDefinition/cumm.git
git clone --depth 1 --branch v2.3.8 https://github.com/traveller59/spconv.git
(cd cumm && pip install -e .)
(cd spconv && pip install -e .)
python -c "import spconv.pytorch as s, torch; print('spconv source build OK', torch.cuda.get_device_name(0))"
