#!/usr/bin/env bash

ex() {
    echo "\$ ${@/eval/}"
    eval "$@"
}

ck() {
    if [ $? != 0 ]; then
        exit 1
    fi
}

# Check privilege
if [ "$(id -u)" != "0" ]; then
    echo "root privilege is required."
    exit 1
fi

ex "nvidia-smi -pm 1"; ck

for i in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
    max_gr=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader,nounits -i $i)
    max_mem=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits -i $i)
    ex "nvidia-smi -ac $max_mem,$max_gr -i $i"; ck
    ex "nvidia-smi -lgc $max_gr -i $i"; ck
done

ex "nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,clocks.max.sm,clocks.max.mem,clocks.max.gr --format=csv"; ck
