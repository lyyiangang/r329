xhost +local:${USER}
docker run --gpus all -it --rm     \
    -e DISPLAY=$DISPLAY     \
    -v /tmp/.X11-unix:/tmp/.X11-unix     \
    -v ${PWD}:${PWD} \
    -w ${PWD} \
    --name zhouyi \
    zepan/zhouyi