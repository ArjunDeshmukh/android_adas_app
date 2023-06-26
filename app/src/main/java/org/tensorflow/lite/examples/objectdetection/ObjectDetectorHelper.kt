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
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorManager
import android.media.AudioManager
import android.media.RingtoneManager
import android.media.ToneGenerator
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.lang.Integer.max
import java.lang.Integer.min


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
    private var cell_phone_width_prev: Float? = null

    // Previous timestamp in milliseconds
    private var previousTimeStamp: Long? = null


    private var sensorManager: SensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    
    private var sensorlistenerobject = SensorListenerClass()

    //private val defaultSoundUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
    //private val ringtone = RingtoneManager.getRingtone(context, defaultSoundUri)

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
                MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
                MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
                MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
                MODEL_MOBILEOBJECTLOCALV1 -> "mobile_object_localizer_v1_1_metadata_2.tflite"
                MODEL_MOBILENETV1_FP32 -> "mobilenet_v1_100_320_fp32_default_1.tflite"
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
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        objectDetectorListener?.onResults(
            results,
            inferenceTime,
            tensorImage.height,
            tensorImage.width)

        findAnObject(results, "person")

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

    private fun findAnObject(results:  MutableList<Detection>?, obj: String){
        // Find if specified object is present. If yes, take a picture
        var cell_phone_width_meas: Float
        var cell_phone_width: Float
        var width_filt_coeff: Float = 0.5F
        var cell_phone_width_chng_ratio: Float = 0.0F
        var f_cell_detected: Boolean = false
        var timeGap: Long = 0L
        var timeToCollision: Float = 0.0F

        if (results != null) {
            for (result in results){
                for (category in result.categories){
                    if (category.label == obj){
                        f_cell_detected = true
                        if(!f_pic_taken) {
                            objectDetectorListener?.takePhoto()
                            f_pic_taken = true
                        }

                        cell_phone_width = result.boundingBox.right - result.boundingBox.left

                        if(cell_phone_width_prev != null && cell_phone_width_prev != 0.0F){
                            //cell_phone_width = cell_phone_width_prev?.plus(width_filt_coeff*(cell_phone_width_meas - cell_phone_width_prev!!))!!
                            cell_phone_width_chng_ratio = cell_phone_width/ cell_phone_width_prev!!
                        }

                        if(cell_phone_width_chng_ratio > 1.0F)
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
                            Log.i("ObjectDetectorHelper", "Cell Phone Width: $cell_phone_width , Count: $obj_width_inc_cnt")
                        }

                        val currentInstant: java.time.Instant = java.time.Instant.now()
                        val currentTimeStamp: Long = currentInstant.toEpochMilli()
                        if(previousTimeStamp != null){
                            timeGap = currentTimeStamp - previousTimeStamp!!
                        }

                        timeToCollision = if(f_obj_width_inc_persistent){
                            timeGap.toFloat()*MILLISEC_TO_SEC/(cell_phone_width_chng_ratio - 1.0F)
                        } else{
                            INFINITY
                        }

                        //Log.i("ObjectDetectorHelper",
                        //    "Width ratio of object: $cell_phone_width_chng_ratio, Time To Collision of $obj is $timeToCollision, Time Gap between detections: " +
                        //            "$timeGap ms"
                        //)


                        if(timeToCollision < 2.0F)
                        {
                            toneGen1.startTone(ToneGenerator.TONE_CDMA_PIP,150);
                            Log.i("ObjectDetectorHelper","Width ratio of object: $cell_phone_width_chng_ratio, Time To Collision of $obj is $timeToCollision, Time Gap between detections: " +
                                        "$timeGap ms")
                        }
                        else
                        {
                        }

                        cell_phone_width_prev = cell_phone_width
                        previousTimeStamp = currentTimeStamp

                    }
                    if(f_cell_detected){break}
                }
                if(f_cell_detected){break}
            }
        }

        if(!f_cell_detected){
            cell_phone_width_prev = 0.0F
            //Log.i("ObjectDetectorHelper", "Frame Missed")

        }
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
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
        const val MODEL_MOBILEOBJECTLOCALV1 = 4
        const val MODEL_MOBILENETV1_FP32 = 5

        const val MILLISEC_TO_SEC: Float = 1.0E-3F
        const val INFINITY: Float = 1000000F

    }
}

