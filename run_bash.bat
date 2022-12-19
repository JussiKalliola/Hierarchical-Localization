#!/bin/bash

set mypath=%cd%
@echo %mypath%

docker run -it --rm -p 8888:8888 ^
--mount "type=bind,src=%mypath%/datasets/,dst=/app/datasets/" ^
--mount "type=bind,src=%mypath%/outputs/,dst=/app/outputs/" ^
--mount "type=bind,src=%mypath%/hloc/,dst=/app/hloc/" ^
--mount "type=bind,src=%mypath%/notebooks/,dst=/app/notebooks/" ^
--mount "type=bind,src=%mypath%/pairs/,dst=/app/pairs/" ^
--mount "type=bind,src=%mypath%/third_party/PatchNetVLAD/,dst=/app/third_party/PatchNetVLAD/" ^
--gpus 1 ^
--entrypoint bash hloc:latest
