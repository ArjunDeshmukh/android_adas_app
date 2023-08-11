package org.tensorflow.lite.examples.adas

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.examples.adas.TfliteRunMode
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.content.res.AssetManager
import android.content.res.AssetFileDescriptor
import android.graphics.RectF


class TfliteRunner(
    context: Context,
    runmode: TfliteRunMode.Mode,
    inputSize: Int,
    conf_thresh: Float,
    iou_thresh: Float
) {
    val numBytesPerChannel_float = 4
    val numBytesPerChannel_int = 1
    external fun postprocess(
        out1: Array<Array<Array<FloatArray>>>?,
        out2: Array<Array<Array<FloatArray>>>?,
        out3: Array<Array<Array<FloatArray>>>?,
        inputSize: Int,
        conf_thresh: Float,
        iou_thresh: Float
    ): Array<FloatArray>

    private var tfliteInterpreter: Interpreter? = null
    lateinit var runmode: TfliteRunMode.Mode
    var inputSize: Int = 0

    inner class InferenceRawResult(inputSize: Int) {
        var out1: Array<Array<Array<FloatArray>>>
        var out2: Array<Array<Array<FloatArray>>>
        var out3: Array<Array<Array<FloatArray>>>

        init {
            out1 = Array(1) { Array(inputSize / 8) { Array(inputSize / 8) { FloatArray(3 * 85) } } }
            out2 =
                Array(1) { Array(inputSize / 16) { Array(inputSize / 16) { FloatArray(3 * 85) } } }
            out3 =
                Array(1) { Array(inputSize / 32) { Array(inputSize / 32) { FloatArray(3 * 85) } } }
        }
    }

    lateinit var inputArray: Array<Any>
    var outputMap: MutableMap<Int, Any>? = null
    lateinit var rawres: InferenceRawResult
    var conf_thresh: Float = 0.0f
    var iou_thresh: Float = 0.0f

    @Throws(Exception::class)
    fun loadModel(context: Context, runmode: TfliteRunMode.Mode?, inputSize: Int, num_threads: Int) {
        lateinit var options: Interpreter.Options
        options.setNumThreads(num_threads)

        when (runmode) {
            //TfliteRunMode.Mode.NONE_FP32 -> options.setUseNNAPI(true)
            TfliteRunMode.Mode.NONE_FP16 ->                 //TODO:deprecated?
                options.setAllowFp16PrecisionForFp32(true)
            //TfliteRunMode.Mode.NONE_INT8 -> options.setUseXNNPACK(true)
            else->throw RuntimeException("Unknown runmode!")
        }

        val quantized_mode: Boolean = runmode?.let { TfliteRunMode.isQuantizedMode(it) } == true
        val precision_str = if (quantized_mode) "int8" else "fp32"
        val modelname = "yolov5s_" + precision_str + "_" + inputSize.toString() + ".tflite"
        val tflite_model_buf = loadModelFile(context.assets, modelname)
        this.tfliteInterpreter = Interpreter(tflite_model_buf, options)
        }

    @Throws(IOException::class)
    private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(modelFilename)
        val inputStream: FileInputStream = FileInputStream(fileDescriptor.getFileDescriptor())
        val fileChannel = inputStream.channel
        val startOffset: Long = fileDescriptor.getStartOffset()
        val declaredLength: Long = fileDescriptor.getDeclaredLength()
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun setInput(resizedbitmap: Bitmap) {
        val quantized_mode: Boolean = TfliteRunMode.isQuantizedMode(runmode)
        val numBytesPerChannel =
            if (quantized_mode) numBytesPerChannel_int else numBytesPerChannel_float
        val imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * numBytesPerChannel)
        val intValues = IntArray(inputSize * inputSize)
        resizedbitmap.getPixels(
            intValues,
            0,
            resizedbitmap.width,
            0,
            0,
            resizedbitmap.width,
            resizedbitmap.height
        )
        imgData.order(ByteOrder.nativeOrder())
        imgData.rewind()
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                if (quantized_mode) {
                    // Quantized model
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else { // Float model
                    val r = (pixelValue shr 16 and 0xFF) / 255.0f
                    val g = (pixelValue shr 8 and 0xFF) / 255.0f
                    val b = (pixelValue and 0xFF) / 255.0f
                    imgData.putFloat(r)
                    imgData.putFloat(g)
                    imgData.putFloat(b)
                }
            }
        }
        inputArray = arrayOf(imgData)
        outputMap?.set(0, rawres.out1)
        outputMap?.set(1, rawres.out2)
        outputMap?.set(2, rawres.out3)
    }

    private var inference_elapsed = 0
    private var postprocess_elapsed = 0
    val lastElapsedTimeLog: String
        get() = String.format(
            "inference: %dms postprocess: %dms",
            inference_elapsed,
            postprocess_elapsed
        )

    fun runInference(): List<Recognition> {
        val bboxes: MutableList<Recognition> = ArrayList()
        val start = System.currentTimeMillis()
        outputMap?.let { tfliteInterpreter?.runForMultipleInputsOutputs(inputArray, it) }
        val end = System.currentTimeMillis()
        inference_elapsed = (end - start).toInt()

        //float[bbox_num][6]
        //                       (x1, y1, x2, y2, conf, class_idx)
        val bbox_arrs = postprocess(
            rawres.out1,
            rawres.out2,
            rawres.out3,
            inputSize,
            conf_thresh,
            iou_thresh
        )
        val end2 = System.currentTimeMillis()
        postprocess_elapsed = (end2 - end).toInt()
        for (bbox_arr in bbox_arrs) {
            bboxes.add(Recognition(bbox_arr))
        }
        return bboxes
    }

    init {
        this.runmode = runmode
        rawres = InferenceRawResult(inputSize)
        this.inputSize = inputSize
        this.conf_thresh = conf_thresh
        this.iou_thresh = iou_thresh
        loadModel(context, runmode, inputSize, 4)
    }

    fun setConfThresh(thresh: Float) {
        conf_thresh = thresh
    }

    fun setIoUThresh(thresh: Float) {
        iou_thresh = thresh
    }
    //port from TfLite Object Detection example
    /** An immutable result returned by a Detector describing what was recognized.  */
    inner class Recognition(
        bbox_array: FloatArray
    ) {
        private val coco_class_names = arrayOf(
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
        )
        val class_idx: Int
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        //private final String id;
        /*public String getId() {
            return id;
        }*/
        /** Display name for the recognition.  */
        val title: String?

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        val confidence: Float?

        /** Optional location within the source image for the location of the recognized object.  */
        private var location: RectF?

        init {
            val x1 = bbox_array[0]
            val y1 = bbox_array[1]
            val x2 = bbox_array[2]
            val y2 = bbox_array[3]
            //this.id = (int)bbox_array[5];
            val class_id = bbox_array[5].toInt()
            class_idx = class_id
            title = coco_class_names[class_id]
            confidence = bbox_array[4]
            location = RectF(x1, y1, x2, y2)
        }

        fun getLocation(): RectF {
            return RectF(location)
        }

        fun setLocation(location: RectF?) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            /*if (id != null) {
                resultString += "[" + id + "] ";
            }*/if (title != null) {
                resultString += "$title "
            }
            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }
            if (location != null) {
                resultString += location.toString() + " "
            }
            return resultString.trim { it <= ' ' }
        }
    }


    init {
        System.loadLibrary("native-lib")
    }
    
    fun getResizedImage(
        bitmap: Bitmap?,
        inputSize: Int
    ): Bitmap {
        return Bitmap.createScaledBitmap(bitmap!!, inputSize, inputSize, true)
    }

    var coco80_to_91class_map = intArrayOf(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90
    )

    fun get_coco91_from_coco80(idx: Int): Int {
        //assume idx < 80
        return coco80_to_91class_map[idx]
        }

}