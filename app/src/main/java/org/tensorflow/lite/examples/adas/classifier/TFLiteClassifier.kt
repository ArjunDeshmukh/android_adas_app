package org.tensorflow.lite.examples.adas.classifier

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks.call
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

open class TFLiteClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    var isInitialized = false
        private set

    private var gpuDelegate: GpuDelegate? = null

    var labels = ArrayList<String>()

    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0
    private var modelInputSize: Int = 0

    private var inp_scale: Float = 0.0F
    private var inp_zero_point: Int = 0
    private var oup_scale: Float = 0.0F
    private var oup_zero_point: Int = 0

    var trial_index:Int = 1

    fun initialize(): Task<Void> {
        return call(
            executorService
        ) {
            initializeInterpreter()
            null
        }
    }

    @Throws(IOException::class)
    fun initializeInterpreter() {

        val assetManager = context.assets
        val model = loadModelFile(assetManager, "yolov5s-int8.tflite")

        labels = loadLines(context, "coco.txt")
        val options = Interpreter.Options()
        options.numThreads = 2
        //gpuDelegate = GpuDelegate()
        //options.addDelegate(gpuDelegate)
        val interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0).shape()
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        modelInputSize = inputImageWidth * inputImageHeight * CHANNEL_SIZE

        this.interpreter = interpreter

        val inpten: Tensor = this.interpreter!!.getInputTensor(0)
        this.inp_scale = inpten.quantizationParams().scale
        this.inp_zero_point = inpten.quantizationParams().zeroPoint

        val oupten: Tensor = this.interpreter!!.getOutputTensor(0)
        this.oup_scale = oupten.quantizationParams().scale
        this.oup_zero_point = oupten.quantizationParams().zeroPoint

        isInitialized = true
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    fun loadLines(context: Context, filename: String): ArrayList<String> {
        val s = Scanner(InputStreamReader(context.assets.open(filename)))
        val labels = ArrayList<String>()
        while (s.hasNextLine()) {
            labels.add(s.nextLine())
        }
        s.close()
        return labels
    }

    private fun getMaxResult(result: List<Detection>): Int {
        var probability = result[0].getScore()
        var labelIndex = 0
        for (i in result.indices) {
            if (probability < result[i].getScore()) {
                probability = result[i].getScore()
                labelIndex = result[i].getLabelIndex()
            }
        }
        return labelIndex
    }

    fun classify(bitmap: Bitmap, imageRotation: Int): String {

        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }
        val resizedImage =
            Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)
        var matrix: Matrix = Matrix()
        matrix.postRotate(imageRotation.toFloat())
        val rotatedBitmap = Bitmap.createBitmap(
            resizedImage, 0, 0,
            resizedImage.width, resizedImage.height, matrix, true
        )

        val byteBuffer = convertBitmapToByteBuffer(resizedImage)
        val outBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 6300 * 85)
        var outputMap: MutableMap<Int, Any> = HashMap<Int, Any>()
        outputMap[0] = outBuffer

        val startTime = SystemClock.uptimeMillis()
        interpreter?.runForMultipleInputsOutputs(arrayOf(byteBuffer), outputMap)
        val endTime = SystemClock.uptimeMillis()
        var inferenceTime = endTime - startTime


        val outData: ByteBuffer = outputMap[0] as ByteBuffer
        val outputArray = getOutArrayFromOutBuffer(outData)
        val detections = getBoundingBoxesClasses(outputArray)
        //val nms_detections = nms(detections)


        var index = getMaxResult(detections)

        return "Prediction is ${labels[index]}\nInference Time $inferenceTime ms"
    }

    private fun nms(list: ArrayList<Detection>): ArrayList<Detection>? {
        val nmsList: ArrayList<Detection> = ArrayList<Detection>()
        for (k in labels.indices) {
            //1.find max confidence per class
            val pq: PriorityQueue<Detection> = PriorityQueue<Detection>(
                50
            ) { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                rhs.getScore().compareTo(lhs.getScore())
            }
            for (i in list.indices) {
                if (list[i].getLabelIndex() === k) {
                    pq.add(list[i])
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a: Array<Detection?> = arrayOfNulls<Detection>(pq.size)
                val detections: Array<Detection> = pq.toArray(a)
                val max: Detection = detections[0]
                nmsList.add(max)
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection: Detection = detections[j]
                    val b: RectF = detection.getBoundingBox()
                    if (box_iou(max.getBoundingBox(), b) < NMS_THRESHOLD) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsList
    }

    private fun getBoundingBoxesClasses(outputArray: Array<Array<FloatArray>>): ArrayList<Detection> {

        val detections: ArrayList<Detection> = ArrayList<Detection>()

        for (i in 0 until OUTPUT_BOXES) {
            val confidence: Float = outputArray[0][i][4]
            var detectedClass = -1
            var maxClass = 0f
            val classes = FloatArray(labels.size)
            for (c in labels.indices) {
                classes[c] = outputArray[0][i][5 + c]
            }
            for (c in labels.indices) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
            //val confidenceInClass = maxClass * confidence
            val confidenceInClass = maxClass
            if (confidenceInClass > OBJ_MINI_CONFIDENCE) {
                val xPos: Float = outputArray[0][i][0]
                val yPos: Float = outputArray[0][i][1]
                val w: Float = outputArray[0][i][2]
                val h: Float = outputArray[0][i][3]

                val rect = RectF(
                    0f.coerceAtLeast(xPos - w / 2),
                    0f.coerceAtLeast(yPos - h / 2),
                    (inputImageWidth - 1).toFloat().coerceAtMost(xPos + w / 2),
                    (inputImageHeight - 1).toFloat().coerceAtMost(yPos + h / 2)
                )
                detections.add(Detection(rect, Category(context, labels[detectedClass], confidenceInClass)))
            }
        }

        return detections
    }

    private fun getOutArrayFromOutBuffer(outData: ByteBuffer):  Array<Array<FloatArray>>{
        val out = Array<Array<FloatArray>>(1) {
            Array<FloatArray>(OUTPUT_BOXES) {
                FloatArray(labels.size + 5)
            }
        }

        var index: Int = 0

        for (i in 0 until OUTPUT_BOXES) {
            for (j in 0 until labels.size + 5) {
                out[0][i][j] =
                    oup_scale * ((outData.get(index++).toInt() and 0xFF) - oup_zero_point)
            }
            // Denormalize xywh
            for (j in 0..3) {
                out[0][i][j] *= modelInputSize.toFloat()
            }
        }

        return out
    }

    fun classifyAsync(bitmap: Bitmap, imageRotation: Int): Task<String> {
        return call(executorService) { classify(bitmap, imageRotation) }
    }

    fun close() {
        call(
            executorService,
            Callable<String> {
                interpreter?.close()
                if (gpuDelegate != null) {
                    gpuDelegate!!.close()
                    gpuDelegate = null
                }

                Log.d(TAG, "Closed TFLite interpreter.")
                null
            }
        )
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        //byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.rewind()

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until inputImageWidth) {
            for (j in 0 until inputImageHeight) {
                val pixelVal = pixels[pixel++]

                byteBuffer.put(((((pixelVal shr 16) and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
                byteBuffer.put(((((pixelVal shr 8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
                byteBuffer.put((((pixelVal and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
            }
        }
        bitmap.recycle()

        return byteBuffer
    }

    protected open fun box_iou(a: RectF, b: RectF): Float {
        return box_intersection(a, b) / box_union(a, b)
    }

    protected open fun box_intersection(a: RectF, b: RectF): Float {
        val w: Float = overlap(
            (a.left + a.right) / 2, a.right - a.left,
            (b.left + b.right) / 2, b.right - b.left
        )
        val h: Float = overlap(
            (a.top + a.bottom) / 2, a.bottom - a.top,
            (b.top + b.bottom) / 2, b.bottom - b.top
        )
        return if (w < 0 || h < 0) 0.0F else w * h
    }

    protected open fun box_union(a: RectF, b: RectF): Float {
        val i = box_intersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    protected open fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = if (l1 > l2) l1 else l2
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = if (r1 < r2) r1 else r2
        return right - left
    }

    companion object {
        private const val TAG = "TfliteClassifier"
        private const val FLOAT_TYPE_SIZE = 4
        private const val CHANNEL_SIZE = 3
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
        private const val OUTPUT_BOXES = 6300
        private const val OBJ_MINI_CONFIDENCE = 0.2F
        private const val NMS_THRESHOLD = 0.6F
    }
}