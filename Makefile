PY ?= python3
export PYTHONPATH := $(CURDIR)/reference:$(CURDIR)/tools

.PHONY: verify selftest vectors lean cpu-norm paper

verify: selftest vectors lean cpu-norm
	@echo "VERIFY: all CPU checks passed"

selftest:
	cd reference && $(PY) lssab_selftest.py | tail -1
	cd reference && $(PY) lssab8_selftest.py | grep -a "SELFTEST"
	cd reference && $(PY) lssab9_selftest.py | grep -a "LSSAB9-SELFTEST" | grep -q "fail=0" && echo "LSSAB9 SELFTEST: PASS (228 cases)"
	cd reference && $(PY) -c "import moe_mxfp4 as M; M.self_test(); print('MOE-MXFP4 SELFTEST: PASS')"

vectors:
	$(PY) tools/contract_vectors.py --check conformance/contract_vectors.json

lean:
	lake build
	lake exe lockstep-formal check
	$(PY) tools/verify_lean_artifacts.py conformance/lean_contract_artifacts.json
	$(PY) tools/verify_lean_artifacts.py --negative conformance/lean_contract_artifacts.json

cpu-norm:
	$(PY) conformance/xarch26/cpu_norm_check.py conformance/xarch26 conformance/xarch26/digests_h100merge.json

paper:
	cd paper && tectonic -X compile main.tex
