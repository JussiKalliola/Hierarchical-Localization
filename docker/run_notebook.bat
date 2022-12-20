#!/bin/bash

set mypath=%cd%
@echo %mypath%

docker run -it --rm -p 8888:8888 ^
--mount "type=bind,src=%mypath%/datasets/,dst=/app/datasets/" ^
--mount "type=bind,src=%mypath%/outputs/,dst=/app/outputs/" ^
--mount "type=bind,src=%mypath%/hloc/,dst=/app/hloc/" ^
--mount "type=bind,src=%mypath%/notebooks/,dst=/app/notebooks/" ^
--mount "type=bind,src=%mypath%/pairs/,dst=/app/pairs/" ^
--gpus 1 ^
--entrypoint bash hloc:latest -c "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
