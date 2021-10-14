#!/bin/bash

set -e
docker build -t qhduan/onnx-pangu-gen:0.1 .
docker push qhduan/onnx-pangu-gen:0.1

