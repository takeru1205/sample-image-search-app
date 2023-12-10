.PHONY: build run embed

all:
	make build
	make run

build:
	docker build --rm=true -t clip-search .

run:
	docker run -it --env-file=app.env --rm --gpus all --name clip-search -v ./:/app clip-search

embed:
	make build
	docker run -it --env-file=app.env --rm --gpus all --name clip-search -v ./:/app clip-search python util.py
