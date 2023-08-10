package org.tensorflow.lite.examples.adas.objecttracker

import org.tensorflow.lite.support.label.Category

class Metric(private val metric: String = "iou") {
    fun distanceMatrix(tracks: List<DoubleArray>, detections: List<DoubleArray>, trackedStateCategories: List<Category>, detectionCategories: List<Category>): Array<DoubleArray> {
        return when (metric) {
            "iou" -> {
                val tracksArr = tracks.map { it.sliceArray(0 until 4) }.toTypedArray()
                val detectionsArr = detections.map { it.sliceArray(0 until 4) }.toTypedArray()
                iouDistanceMatrix(tracksArr, detectionsArr, trackedStateCategories, detectionCategories)
            }
            "euc" -> {
                eucDistanceMatrix(tracks, detections, trackedStateCategories, detectionCategories)
            }
            else -> throw IllegalArgumentException("Invalid metric")
        }
    }

    private fun iouDistanceMatrix(tracks: Array<DoubleArray>, detections: Array<DoubleArray>, trackedStateCategories: List<Category>, detectionCategories: List<Category>): Array<DoubleArray> {
        val dm = Array(tracks.size) { DoubleArray(detections.size) }
        for (i in tracks.indices) {
            for (j in detections.indices) {
                dm[i][j] = - iou(tracks[i], detections[j]) //Negative sign because larger the iou, lower the cost
                if (trackedStateCategories[i].label != detectionCategories[j].label) //If labels don't match then set cost an infinite
                {
                    dm[i][j] = Double.MAX_VALUE
                }

            }
        }
        return dm
    }

    private fun eucDistanceMatrix(tracks: List<DoubleArray>, detections: List<DoubleArray>, trackedStateCategories: List<Category>, detectionCategories: List<Category>): Array<DoubleArray> {
        val dm = Array(tracks.size) { DoubleArray(detections.size) }
        for (i in tracks.indices) {
            for (j in detections.indices) {
                dm[i][j] = euc(tracks[i], detections[j])
                if (trackedStateCategories[i].label != detectionCategories[j].label) //If labels don't match then set cost an infinite
                {
                    dm[i][j] = Double.MAX_VALUE
                }
            }
        }
        return dm
    }

    private fun mdist(tracks: List<DoubleArray>, detections: List<DoubleArray>, func: (DoubleArray, DoubleArray) -> Double): Array<DoubleArray> {
        val dm = Array(tracks.size) { DoubleArray(detections.size) }
        for (i in tracks.indices) {
            for (j in detections.indices) {
                dm[i][j] = func(tracks[i], detections[j])
            }
        }
        return dm
    }

    private fun iou(boxA: DoubleArray, boxB: DoubleArray): Double {
        val xA = maxOf(boxA[0], boxB[0])
        val yA = maxOf(boxA[1], boxB[1])
        val xB = minOf(boxA[2], boxB[2])
        val yB = minOf(boxA[3], boxB[3])

        val interArea = maxOf(0.0, xB - xA + 1) * maxOf(0.0, yB - yA + 1)

        val boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        val boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        val iou = interArea / (boxAArea + boxBArea - interArea)
        return iou
    }

    private fun euc(arr1: DoubleArray, arr2: DoubleArray): Double {
        var sum = 0.0
        for (i in arr1.indices) {
            val diff = arr1[i] - arr2[i]
            sum += diff * diff
        }
        return Math.sqrt(sum)
    }

}
