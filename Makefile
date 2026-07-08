.PHONY: test test-all smoke viewer soak

test:
	pytest -x -q -m "not slow"

test-all:
	pytest -x -q

smoke:
	python train.py --config configs/smoke.yaml

viewer:
	python -m viz.viewer --live runs/latest

soak:
	python scripts/soak.py
