.PHONY: db-up db-init

db-up:
	docker compose up -d

db-init:
	python -m papertrail.db.repository