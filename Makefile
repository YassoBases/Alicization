.PHONY: test smoke viewer soak

test:
	pytest -x -q

smoke:
	python train.py --config configs/smoke.yaml

viewer:
	python -m viz.viewer --run runs/latest

soak:
	python scripts/soak.py
