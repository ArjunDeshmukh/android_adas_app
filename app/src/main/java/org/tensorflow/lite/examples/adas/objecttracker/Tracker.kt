package org.tensorflow.lite.examples.adas.objecttracker

import android.util.Log
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.MatrixUtils
import java.util.LinkedHashMap
import org.apache.commons.math3.linear.RealVector
import org.tensorflow.lite.support.label.Category
import kotlin.math.abs

open class Tracker {
    /**
     * Parent class for the general Tracker case, intended for creating the basis for inheritance for specialized trackers.
     * Assumes the default use of KalmanFilter to assist tracking.
     */
    private var nextTrackID = 0
    private var matchingThreshold: Double? = null
    val tracked = LinkedHashMap<Int, KalmanTrack>()
    private val disappeared = LinkedHashMap<Int, Int>()
    private val numContDetections = LinkedHashMap<Int, Int>()
    private var maxDisappeared = 3
    private var matureContDetScans = 3
    private var metric: Metric?
    private var objConfThreshold: Double = 0.50

    constructor(metric: Metric?, matchingThreshold: Double?, maxDisappeared: Int, objConfThreshold: Double) {
        this.metric = metric
        this.matchingThreshold = matchingThreshold
        this.maxDisappeared = maxDisappeared
        this.objConfThreshold = objConfThreshold
    }

    fun register(state: RealVector?, category: Category) {
        this.tracked[this.nextTrackID] = KalmanTrack(state!!.toArray(), category)
        this.disappeared[this.nextTrackID] = 0
        this.numContDetections[this.nextTrackID] = 1
        this.nextTrackID++
    }

    fun deregister(objectID: Int) {
        this.tracked.remove(objectID)
        this.disappeared.remove(objectID)
        this.numContDetections.remove(objectID)
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

    fun project(): List<Triple<Int, RealVector, Category>> {
        val tracks = ArrayList<Triple<Int, RealVector, Category>>()
        for ((ID, track) in this.tracked) {
            if(track.getObjectStatus() != TrackStatus.NEW){
                tracks.add(Triple(ID, MatrixUtils.createRealVector(track.project()), track.getCategory()))
            }
        }

        val objectsList = ArrayList<ObjectClass>()
        for ((ID, track) in this.tracked) {
            var obj = ObjectClass()
            obj.ID = ID
            obj.top_left_bottom_right_coord = MatrixUtils.createRealVector(track.project())
            obj.category = track.getCategory()
            obj.filtWidth = track.filtWidth()
            objectsList.add(obj)
        }

        return tracks
    }

    fun linearAssignment(D: Array<DoubleArray>, trackIDs: List<Int>, detections: List<RealVector>, categories: List<Category>) {

        val assignedRowsCols = greedyAssignment(D)
        val usedRows = HashSet<Int>()
        val usedCols = HashSet<Int>()
        for (row_and_col in assignedRowsCols) {
            val row = row_and_col.first
            val col = row_and_col.second
            if (row in usedRows || col in usedCols) {
                continue
            } else if (abs(D[row][col])  > this.matchingThreshold!!) {
                val trackID = trackIDs[row]
                this.tracked[trackID] = this.tracked[trackID]!!.update(detections[col])
                this.disappeared[trackID] = 0
                this.numContDetections[trackID] = this.numContDetections[trackID]!! + 1
                usedRows.add(row)
                usedCols.add(col)
            }
        }
        val unusedRows = D.indices.toHashSet().subtract(usedRows)
        val unusedCols = D[0].indices.toHashSet().subtract(usedCols)
        if (D.size >= D[0].size) {
            for (row in unusedRows) {
                val objectID = trackIDs[row]
                this.disappeared[objectID] = this.disappeared[objectID]!! + 1
                this.numContDetections[objectID] = 0
                if (this.disappeared[objectID]!! > this.maxDisappeared) {
                    this.deregister(objectID)
                }
            }
        } else {
            for (col in unusedCols) {
                if (categories[col].score >= this.objConfThreshold)
                    this.register(detections[col], categories[col])
            }
        }
    }

    fun reset() {
        this.nextTrackID = 0
        this.tracked.clear()
        this.disappeared.clear()
        this.numContDetections.clear()
    }

    fun statusUpdate() {
        for(key in this.tracked.keys){
            if(this.tracked[key]?.getObjectStatus()  == TrackStatus.NEW){

                //Log.i("Tracker", "Track key: $track.key ")
                if(this.numContDetections[key]!! >= this.matureContDetScans )
                {
                    this.tracked[key]?.setObjectStatus(TrackStatus.MATURE)
                }
            }
            if(this.tracked[key]?.getObjectStatus()  == TrackStatus.MATURE){
                if(this.disappeared[key]!! > 0){
                    this.tracked[key]?.setObjectStatus(TrackStatus.COASTED)
                }
            }
            if(this.tracked[key]?.getObjectStatus() == TrackStatus.COASTED){
                if(this.disappeared[key]!! == 0) {
                    this.tracked[key]?.setObjectStatus(TrackStatus.MATURE)
                }
            }
        }
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

class KalmanTracker(private val metric: Metric = Metric("iou"), private val matchingThreshold: Double = 0.2) : Tracker(null, matchingThreshold, 10, 0.70) {
    /**
     * Specialized tracker class which inherits from the basic Tracker class
     * Utilizes the KalmanFilter and the IoU metric for more robust bounding box associations
     */
    fun update(detections: List<RealVector>, categories: List<Category>): List<Triple<Int, RealVector, Category>> {
        if (detections.isEmpty()) {
            return this.handleNoDetections().map { Triple(it.key, MatrixUtils.createRealVector(it.value.project()), it.value.getCategory()) }
        }
        if (this.tracked.isEmpty()) {
            for (i in detections.indices) {
                this.register(detections[i], categories[i])
            }
        } else {
            this.associate(  detections.map { vector -> vector.toArray() }.toTypedArray(), categories)
        }
        return this.project()
    }

    private fun associate(detections: Array<DoubleArray>, detectionCategories: List<Category>) {
        // Grab the set of object IDs and corresponding states
        val trackIds = tracked.keys.toList()

        // Get predicted tracked object states from KalmanFilter
        val trackedStates = tracked.values.map { track -> track.predict() }

        val trackedStateCategories: List<Category> = tracked.values.map { track -> track.getCategory() }

        // Compute the distance matrix between detections and trackers according to metric
        val distanceMatrix = metric.distanceMatrix(trackedStates.toList(), detections.toList(), trackedStateCategories, detectionCategories)

        // Associate detections to existing trackers according to distance matrix
        linearAssignment(distanceMatrix, trackIds, detections.map { row -> ArrayRealVector(row)}, detectionCategories)

        // Update status of each tracked object
        statusUpdate()
    }


    fun calcFiltWidths(): List<Float> {
        return tracked.values.map { track -> track.filtWidth() }
    }

}


