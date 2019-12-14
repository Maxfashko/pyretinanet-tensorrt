### Python-обертка над NN tensorrt-retinanet

#### Сборка

```
git submodule init
git submodule update
```

Для Ubuntu 18.04
```
mkdir cmake-build-debug
cd cmake-build-debug
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++11" -DCMAKE_INSTALL_PREFIX=/usr -DPYTHON_VERSION="3.5" -DBOOST_PYTHON_VERSION="python-py35" ..
```
