package org.tensorflow.lite.examples.adas

object TfliteRunMode {
    fun isQuantizedMode(mode: Mode): Boolean {
        return mode == Mode.NONE_INT8 || mode == Mode.NNAPI_DSP_INT8
    }

    enum class Mode {
        NONE_FP32, NONE_FP16, NONE_INT8, NNAPI_GPU_FP32, NNAPI_GPU_FP16, NNAPI_DSP_INT8
    }
}