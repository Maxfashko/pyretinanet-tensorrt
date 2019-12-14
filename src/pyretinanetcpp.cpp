// pytsm
#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <string>
#include <memory>
#include <chrono>

#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "tensorrt/engine.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

typedef std::vector<float> MyList;


class RetinaNet{
public:
    vector<float> mean {0.485, 0.456, 0.406};
    vector<float> std {0.229, 0.224, 0.225};
    int channels = 3;
    retinanet::Engine *engine;
    int num_det;
    vector<int> inputSize;
    std::vector<float> detected_object;
    float threshold = 0.5;

    RetinaNet(char *engine_path, float threshold){
        this->threshold = threshold;

        std::cout << "Loading engine..." << std::endl;
        this->engine = new retinanet::Engine(engine_path);
        this->inputSize = engine->getInputSize();
        this->num_det = engine->getMaxDetections();
    }

    vector<float> preprocess_image(cv::Mat &image){
        cv::resize(image, image, cv::Size(this->inputSize[1], this->inputSize[0]));
        cv::Mat pixels;
        image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);
        vector<float> img;
        vector<float> data (this->channels * this->inputSize[0] * this->inputSize[1]);

        if (pixels.isContinuous())
            img.assign((float*)pixels.datastart, (float*)pixels.dataend);
        else {
            cerr << "Error reading image " << endl;
            return data;
        }

        for (int c = 0; c < this->channels; c++) {
            for (int j = 0, hw = this->inputSize[0] * this->inputSize[1]; j < hw; j++) {
                data[c * hw + j] = (img[this->channels * j + 2 - c] - this->mean[c]) / this->std[c];
            }
        }
        return data;
    }

    MyList detect(PyObject *o){

        // auto start = chrono::steady_clock::now();

        if (!o || o == Py_None) {
            std::vector<float> crd{};
            return crd;
        }

        if (!PyArray_Check(o) ) {
            throw std::invalid_argument("Is not a numpy array");
        }

        cv::Mat image = pbcvt::fromNDArrayToMat(o);

        vector<float> data = this->preprocess_image(image);

        if (!data.size() > 0){
          std::vector<float> crd{};
          return crd;
        }

        // Create device buffers
        void *data_d, *scores_d, *boxes_d, *classes_d;
        cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
        cudaMalloc(&scores_d, num_det * sizeof(float));
        cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
        cudaMalloc(&classes_d, num_det * sizeof(float));

        // Copy image to device
        size_t dataSize = data.size() * sizeof(float);
        cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

        vector<void *> buffers = { data_d, scores_d, boxes_d, classes_d };
        engine->infer(buffers, 1);

        // Get back the bounding boxes
        auto scores = new float[num_det];
        auto boxes = new float[num_det * 4];
        auto classes = new float[num_det];
        cudaMemcpy(scores, scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
        cudaMemcpy(boxes, boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(classes, classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);

        if (detected_object.size() > 0){
            detected_object.clear();
        }

        for (int i = 0; i < num_det; i++) {
            if (scores[i] >= this->threshold) {
                float x1 = boxes[i * 4 + 0];
                float y1 = boxes[i * 4 + 1];
                float x2 = boxes[i * 4 + 2];
                float y2 = boxes[i * 4 + 3];

                detected_object.push_back(classes[i]);
                detected_object.push_back(scores[i]);
                detected_object.push_back(x1);
                detected_object.push_back(y1);
                detected_object.push_back(x2);
                detected_object.push_back(y2);
            }
        }

        delete[] scores;
        delete[] boxes;
        delete[] classes;
        cudaFree(data_d);
        cudaFree(scores_d);
        cudaFree(boxes_d);
        cudaFree(classes_d);

        // auto end = std::chrono::steady_clock::now();
        // auto diff = end - start;

        // std::cout << "cpp time exec:" << std::chrono::duration <double, milli> (diff).count() << " ms" << std::endl;

        return detected_object;
    }
};


static void* init_python_lib()
{
    Py_Initialize();
    import_array();
}

static void translate_invalid_argument(std::invalid_argument const &e)
{
    PyErr_SetString(PyExc_ValueError, e.what());
}

static void* init()
{
    Py_Initialize();
    import_array();
}


BOOST_PYTHON_MODULE(pyretinanetcpp)
{
    init();

    pbcvt::matFromNDArrayBoostConverter();

    boost::python::class_<MyList>("MyList")
        .def(boost::python::vector_indexing_suite<MyList>() );

    boost::python::class_<RetinaNet>("RetinaNet", boost::python::init<char *, float>())
            .def("detect",  &RetinaNet::detect);

    boost::python::register_exception_translator<std::invalid_argument>(translate_invalid_argument);
}
