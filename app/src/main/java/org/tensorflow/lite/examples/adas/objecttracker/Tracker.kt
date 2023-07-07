package org.tensorflow.lite.examples.adas.objecttracker

import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.MatrixUtils
import java.util.LinkedHashMap
import org.apache.commons.math3.linear.RealVector


open class Tracker {
    /**
     * Parent class for the general Tracker case, intended for creating the basis for inheritance for specialized trackers.
     * Assumes the default use of KalmanFilter to assist tracking.
     */
    private var nextTrackID = 0
    private var matchingThreshold: Double? = null
    val tracked = LinkedHashMap<Int, KalmanTrack>()
    private val disappeared = LinkedHashMap<Int, Int>()
    private var maxDisappeared = 10
    private var metric: Metric?

    constructor(metric: Metric?, matchingThreshold: Double?, maxDisappeared: Int) {
        this.metric = metric
        this.matchingThreshold = matchingThreshold
        this.maxDisappeared = maxDisappeared
    }

    fun register(state: RealVector?) {
        this.tracked[this.nextTrackID] = KalmanTrack(state!!.toArray())
        this.disappeared[this.nextTrackID] = 0
        this.nextTrackID++
    }

    fun deregister(objectID: Int) {
        this.tracked.remove(objectID)
        this.disappeared.remove(objectID)
    }

    fun handleNoDetections(): LinkedHashMap<Int, KalmanTrack> {
        for (trackID in this.disappeared.keys) {
            this.disappeared[trackID] = this.disappeared[trackID]!! + 1
            if (this.disappeared[trackID]!! > this.maxDisappeared) {
                this.deregister(trackID)
            }
        }
        return this.tracked
    }

    fun project(): List<Pair<Int, RealVector>> {
        val tracks = ArrayList<Pair<Int, RealVector>>()
        for ((ID, track) in this.tracked) {
            tracks.add(ID to MatrixUtils.createRealVector(track.project()))
        }
        return tracks
    }

    fun linearAssignment(D: Array<DoubleArray>, trackIDs: List<Int>, detections: List<RealVector>) {
        val negativeD = Array(D.size) { row ->
            DoubleArray(D[row].size) { col ->
                -D[row][col]
            }
        }
        val (rows, cols) = linearSumAssignment(negativeD)
        val usedRows = HashSet<Int>()
        val usedCols = HashSet<Int>()
        for ((row, col) in rows.zip(cols)) {
            if (row in usedRows || col in usedCols) {
                continue
            } else if (D[row][col] > this.matchingThreshold!!) {
                val trackID = trackIDs[row]
                this.tracked[trackID] = this.tracked[trackID]!!.update(detections[col])
                this.disappeared[trackID] = 0
                usedRows.add(row)
                usedCols.add(col)
            }
        }
        val unusedRows = setOf(0 until D.size).subtract(usedRows)
        val unusedCols = setOf(0 until D[0].size).subtract(usedCols)
        if (D.size >= D[0].size) {
            for (row in unusedRows) {
                val objectID = trackIDs[row as Int]
                this.disappeared[objectID] = this.disappeared[objectID]!! + 1
                if (this.disappeared[objectID]!! > this.maxDisappeared) {
                    this.deregister(objectID)
                }
            }
        } else {
            for (col in unusedCols) {
                this.register(detections[col as Int])
            }
        }
    }

    fun reset() {
        this.nextTrackID = 0
        this.tracked.clear()
        this.disappeared.clear()
    }

    companion object {
        fun cropBBoxFromFrame(frame: Array<Array<Double>>, bboxes: List<RealVector>): List<Array<Array<Double?>>> {
            val bboxesCrop = ArrayList<Array<Array<Double?>>>()
            for (bbox in bboxes) {
                val x1 = bbox.getEntry(0).toInt()
                val y1 = bbox.getEntry(1).toInt()
                val x2 = bbox.getEntry(2).toInt()
                val y2 = bbox.getEntry(3).toInt()
                val bboxCrop = Array(y2 - y1 + 1) { arrayOfNulls<Double>(x2 - x1 + 1) }
                for (i in y1..y2) {
                    for (j in x1..x2) {
                        bboxCrop[i - y1][j - x1] = frame[i][j]
                    }
                }
                bboxesCrop.add(bboxCrop)
            }
            return bboxesCrop
        }
    }
}

class KalmanTracker(private val metric: Metric = Metric("iou"), private val matchingThreshold: Double = 0.2) : Tracker(null, matchingThreshold, 10) {
    /**
     * Specialized tracker class which inherits from the basic Tracker class
     * Utilizes the KalmanFilter and the IoU metric for more robust bounding box associations
     */
    fun update(detections: List<RealVector>): List<Pair<Int, RealVector>> {
        if (detections.isEmpty()) {
            return this.handleNoDetections().map { it.key to MatrixUtils.createRealVector(it.value.project()) }
        }
        if (this.tracked.isEmpty()) {
            for (i in detections.indices) {
                this.register(detections[i])
            }
        } else {
            this.associate(  detections.map { vector -> vector.toArray() }.toTypedArray())
        }
        return this.project()
    }

    private fun associate(detections: Array<DoubleArray>) {
        // Grab the set of object IDs and corresponding states
        val trackIds = tracked.keys.toList()

        // Get predicted tracked object states from KalmanFilter
        val trackedStates = tracked.values.map { track -> track.predict() }

        // Compute the distance matrix between detections and trackers according to metric
        val distanceMatrix = metric.distanceMatrix(trackedStates.toList(), detections.toList())

        // Associate detections to existing trackers according to distance matrix
        linearAssignment(distanceMatrix, trackIds, detections.map { row -> ArrayRealVector(row)})
    }

}


