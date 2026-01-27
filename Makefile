.PHONY: test stability run docker-build docker-run

test:
	python -m pytest -q

stability:
	python -m wrinkle_v2_improvements.stability_eval --input_dir sample_images --out_dir stability_reports --config configs/default.yaml

run:
	uvicorn service.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -f Dockerfile.service -t wrinkle-v2-service .

docker-run:
	docker run --rm -p 8000:8000 wrinkle-v2-service
