/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.adas

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.hardware.Sensor
import android.hardware.SensorManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.SystemClock
import android.util.Log
import org.apache.commons.math3.linear.MatrixUtils
import org.tensorflow.lite.examples.adas.objecttracker.KalmanTracker
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.lang.Integer.min
import org.apache.commons.math3.linear.RealVector
import org.tensorflow.lite.support.label.Category
import kotlin.math.sqrt


class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null

    // Flag to indicate if one picture has been taken
    private var f_pic_taken: Boolean = false

    // Previous time step cell phone bounding box width
    private var obj_width_prev: Float? = null

    private var count: Int = 0
    private var mean: Float = 0.0F
    private var m2: Float = 0.0F

    // Previous timestamp in milliseconds
    private var previousTimeStamp: Long? = null


    private var sensorManager: SensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    
    private var sensorlistenerobject = SensorListenerClass()

    //private val defaultSoundUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
    //private val ringtone = RingtoneManager.getRingtone(context, defaultSoundUri)
    private var tracker = KalmanTracker(matchingThreshold = 0.2, maxDisappeared = 3, objConfThreshold = 0.6)

    var toneGen1 = ToneGenerator(AudioManager.STREAM_MUSIC, 100)

    private var f_obj_width_inc_persistent: Boolean = false
    private var obj_width_inc_cnt: Int = 0
    private val k_obj_width_inc_cnt_thresh: Int = 3

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null

        sensorManager.unregisterListener(sensorlistenerobject)
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    private fun setupObjectDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
            when (currentModel) {
                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                MODEL_MOBILENETV2_640 -> "mobilenetv2_640x640.tflite"
                MODEL_MOBILEOBJECTLOCALV1 -> "mobile_object_localizer_v1_1_metadata_2.tflite"
                else -> "mobilenetv1.tflite"
            }

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details"
            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }

        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.also { accelerometer ->
            sensorManager.registerListener(
                sensorlistenerobject,
                accelerometer,
                SensorManager.SENSOR_DELAY_NORMAL,
                SensorManager.SENSOR_DELAY_UI
            )
        }

        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)?.also { magneticField ->
            sensorManager.registerListener(
                sensorlistenerobject,
                magneticField,
                SensorManager.SENSOR_DELAY_NORMAL,
                SensorManager.SENSOR_DELAY_UI
            )
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector?.detect(tensorImage)
        val (tracks, filtWidths) = trackObjects(results)
        val trackedResults = trackerOPtoModelOP(tracks)


        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        objectDetectorListener?.onResults(
            trackedResults,
            inferenceTime,
            tensorImage.height,
            tensorImage.width)

        findAnObject(trackedResults, "car", filtWidths)

        /*
        * val orientationAngles = sensorlistenerobject.getOrientationAngles()
        val accelerometerReading = sensorlistenerobject.getAccelerometerReading()

        // "Orientation Angles: ${orientationAngles[0]}, ${orientationAngles[1]}, ${orientationAngles[2]}," +
        //"Accelerometer Reading: ${accelerometerReading[0]}, ${accelerometerReading[1]}, ${accelerometerReading[2]}"

        Log.i("ObjectDetectorHelper",
            "Orientation Angles: ${orientationAngles[0]}, ${orientationAngles[1]}, ${orientationAngles[2]}"
        )
        * */



    }

    private fun findAnObject(results:  MutableList<Detection>?, obj: String, filtWidths: List<Float?>){
        // Find if specified object is present. If yes, take a picture
        var obj_width: Float?
        var obj_width_chng_ratio: Float = 0.0F
        var f_obj_detected: Boolean = false
        var timeGap: Long = 0L
        var timeToCollision: Float = 0.0F

        if (results != null) {
            for (i in 0 until results.size){
                var result = results[i]
                for (category in result.categories){
                    if (category.label == obj){
                        f_obj_detected = true
                        if(!f_pic_taken) {
                            objectDetectorListener?.takePhoto()
                            f_pic_taken = true
                        }

                        obj_width = filtWidths[i]

                        if(obj_width != null)
                        {
                            if(obj_width_prev != null && obj_width_prev != 0.0F){
                                obj_width_chng_ratio = obj_width/ obj_width_prev!!
                                count++
                                val delta = obj_width - mean
                                mean += delta / count
                                val delta2 = obj_width - mean
                                m2 += delta*delta2
                            }
                            else
                            {
                                count = 1
                                mean = obj_width
                                m2 = 0.0F
                            }

                        }


                        val standardDeviation: Float = if (count < 2) Float.NaN else sqrt(m2 / (count - 1).toFloat())

                        if(obj_width_chng_ratio > 1.0F)
                        {
                            obj_width_inc_cnt += 1
                            obj_width_inc_cnt = min(obj_width_inc_cnt, 1000)
                        }
                        else
                        {
                            obj_width_inc_cnt = 0
                        }

                        f_obj_width_inc_persistent = obj_width_inc_cnt >= k_obj_width_inc_cnt_thresh

                        if(f_obj_width_inc_persistent)
                        {
                            //Log.i("ObjectDetectorHelper", "Cell Phone Width: $obj_width , Count: $obj_width_inc_cnt")
                        }

                        val currentInstant: java.time.Instant = java.time.Instant.now()
                        val currentTimeStamp: Long = currentInstant.toEpochMilli()
                        if(previousTimeStamp != null){
                            timeGap = currentTimeStamp - previousTimeStamp!!
                        }

                        timeToCollision = if(f_obj_width_inc_persistent){
                            timeGap.toFloat()*MILLISEC_TO_SEC/(obj_width_chng_ratio - 1.0F)
                        } else{
                            INFINITY
                        }

                        //Log.i("ObjectDetectorHelper",
                        //    "Width ratio of object: $obj_width_chng_ratio, Time To Collision of $obj is $timeToCollision, Time Gap between detections: " +
                        //            "$timeGap ms"
                        //)
                        Log.i("ObjectDetectorHelper", "Obj Width: $obj_width , Mean: $mean, Std Dev: $standardDeviation" +
                               "Time Gap: $timeGap")



                        if(timeToCollision < 2.0F)
                        {
                            toneGen1.startTone(ToneGenerator.TONE_CDMA_PIP,150);
                            //Log.i("ObjectDetectorHelper","Width ratio of object: $obj_width_chng_ratio, Time To Collision of $obj is $timeToCollision, Time Gap between detections: " +
                            //           "$timeGap ms")
                        }
                        else
                        {
                        }

                        obj_width_prev = obj_width
                        previousTimeStamp = currentTimeStamp

                    }
                    if(f_obj_detected){break}
                }
                if(f_obj_detected){break}
            }
        }

        if(!f_obj_detected){
            obj_width_prev = 0.0F
            //Log.i("ObjectDetectorHelper", "Frame Missed")

        }
    }

    private fun trackObjects(results: MutableList<Detection>?):  Pair<List<Triple<Int, RealVector, Category>>, List<Float?>>{

        val (detections, categories) = modelOPtoTrackerIP(results)

        val tracks: List<Triple<Int, RealVector, Category>> = tracker.update(detections, categories)

        val filtWidths: List<Float?> = tracker.calcFiltWidths()

        return Pair(tracks, filtWidths)

    }

    private fun modelOPtoTrackerIP(results: MutableList<Detection>?): Pair<MutableList<RealVector>, MutableList<Category>> {

        var detections: MutableList<RealVector> = listOf<RealVector>().toMutableList()
        var categories: MutableList<Category> = listOf<Category>().toMutableList()

        if (results != null) {
            for (result in results) {
                detections.add(MatrixUtils.createRealVector(doubleArrayOf(result.boundingBox.left.toDouble(), result.boundingBox.top.toDouble(),
                    result.boundingBox.right.toDouble(), result.boundingBox.bottom.toDouble())))
                categories.add(result.categories[0])
            }
        }

        return Pair(detections, categories)
    }

    private fun trackerOPtoModelOP(tracks: List<Triple<Int, RealVector, Category>>): MutableList<Detection>? {

        var trackedResults: MutableList<Detection>? = mutableListOf()

        for (track in tracks)
        {
            var tempTrackedResult: Detection = Detection.create(RectF(track.second.getEntry(0).toFloat(), track.second.getEntry(1).toFloat(),
                track.second.getEntry(2).toFloat(), track.second.getEntry(3).toFloat()), mutableListOf(track.third))

            trackedResults?.add(tempTrackedResult)

        }

        return trackedResults

    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
          results: MutableList<Detection>?,
          inferenceTime: Long,
          imageHeight: Int,
          imageWidth: Int
        )

        fun takePhoto()
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_MOBILENETV2_640 = 1
        const val MODEL_MOBILEOBJECTLOCALV1 = 2

        const val MILLISEC_TO_SEC: Float = 1.0E-3F
        const val INFINITY: Float = 1000000F

    }
}

