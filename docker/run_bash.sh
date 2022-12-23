#!/bin/bash

export SCRIPT=$(readlink -f "$0")
export CWD=$(dirname "$SCRIPT")

xhost + local:
docker run \
	-it --rm \
	-p 8888:8888 \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--runtime=nvidia \
	--mount "type=bind,src=$CWD/../datasets/,dst=/app/datasets/" \
	--mount "type=bind,src=$CWD/../outputs/,dst=/app/outputs/" \
	--mount "type=bind,src=$CWD/../hloc/,dst=/app/hloc/" \
	--mount "type=bind,src=$CWD/../notebooks/,dst=/app/notebooks/" \
	--mount "type=bind,src=$CWD/../pairs/,dst=/app/pairs/" \
	--mount "type=bind,src=$CWD/../third_party/PatchNetVLAD/,dst=/app/third_party/PatchNetVLAD/" \
	--gpus 'all,"capabilities=graphics,utility,display,video,compute"' \
	--entrypoint /bin/bash hloc:latest
xhost - local: 
