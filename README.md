# Python wrapper over NN tensorrt-retinanet

Check out this repository (https://github.com/NVIDIA/retinanet-examples) to find out how to get weights for the model.

### Build

```
git submodule update --init --recursive
```

#### Requirements
Jetson TX2/XAVIER with JetPack 4.2.3 Developer Preview (CUDA 10, cuDNN 7.3.1, TensorRT 5.0.3, OpenCV 3.3.1)

```bash
sudo apt-get update
sudo apt-get install -y gcc g++ vim
sudo apt-get install -y libpython3-dev
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y libopencv-dev
sudo apt-get install -y python-opencv python3-numpy
sudo apt-get install -y python3-pip

sudo pip3 install --upgrade pip
sudo pip3 install setuptools
sudo apt-get install -y libpython3-all-dev
```

make
```
mkdir cmake-build-debug
cd cmake-build-debug
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++11" -DPYTHON_VERSION="3.6" -DBOOST_PYTHON_VERSION="python-py36" .. && sudo make -j4 install
```

### Run

```bash
python3 example/inference.py
```

### Convert weights

You can get onnx weights from this [link](https://yadi.sk/d/oBk0NIInBrA-pQ). Weights were obtained by training on the semantic drone dataset https://www.tugraz.at/index.php?id=22387
To convert weights, use [cpp api](https://github.com/NVIDIA/retinanet-examples/tree/master/extras/cppapi).

```bash
./export retinanet_rn50.onnx retinanet_rn50.plan
```
