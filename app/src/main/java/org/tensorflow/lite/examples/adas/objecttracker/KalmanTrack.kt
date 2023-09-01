package org.tensorflow.lite.examples.adas.objecttracker

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealVector
import org.tensorflow.lite.examples.adas.SecondOrderFilter
import org.tensorflow.lite.support.label.Category
import kotlin.math.sqrt

class KalmanTrack(initial_state: DoubleArray, category: Category) {
    private val kf: KalmanFilter = KalmanFilter(dim_x = 7, dim_z = 4)

    init {
        // Transition matrix
        kf.F = MatrixUtils.createRealMatrix(arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        ))

        // Transformation matrix (Observation to State)
        kf.H = MatrixUtils.createRealMatrix(arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        ))

        kf.R.setSubMatrix(arrayOf(
            doubleArrayOf(10.0, 0.0),
            doubleArrayOf(0.0, 10.0)
        ), 2, 2) // observation error covariance


        kf.P.setSubMatrix(kf.P.getSubMatrix(4, 6, 4, 6).scalarMultiply(1000.0).data,
            4, 4)  // initial velocity error covariance
        kf.P = kf.P.scalarMultiply(10.0) // initial location error covariance

        kf.Q = kf.Q.scalarMultiply(10.0)
        kf.Q.multiplyEntry(6, 6, 0.01) // process noise
        kf.Q.setSubMatrix(kf.Q.getSubMatrix(4, 6, 4, 6).scalarMultiply(0.01).data,
            4, 4) // process noise

        kf.x.setSubVector(0, MatrixUtils.createRealVector(xxyyToXysr(initial_state)))    // initialize KalmanFilter state
    }

    private val objCategory = category
    private var objStatus = TrackStatus.NEW
    private val secondOrderFilter: SecondOrderFilter = SecondOrderFilter(25.0F, 5.0F)

    private var disappearedScans: Int = 0

    fun getDisappeared(): Int {
        return disappearedScans
    }
    fun incrementDisappeared(){
        disappearedScans++
    }

    fun resetDisappeared(){
        disappearedScans = 0
    }

    private var numContDetections: Int = 1
    fun getContDet(): Int {
        return numContDetections
    }
    fun incrementContDet(){
        numContDetections++
    }
    fun resetContDet(){
        numContDetections = 0
    }

    fun project(): DoubleArray {
        return xysrToXxyy( kf.x.toArray())
    }

    fun update(newDetection: RealVector): KalmanTrack {
        kf.update(MatrixUtils.createRealVector(xxyyToXysr(newDetection.toArray())))
        return this
    }

    fun predict(): DoubleArray {
        if (kf.x.getEntry(6) + kf.x.getEntry(6) <= 0) {
            kf.x.setEntry(6, 0.0)
        }
        kf.predict()
        return project()
    }

    fun getCategory(): Category{
        return objCategory
    }


    fun filtWidth(): Float? {
        if (objStatus != TrackStatus.NEW)
        {
            return secondOrderFilter.process(sqrt((kf.x.toArray()[2] * kf.x.toArray()[3]).toFloat()))
        }
        else
        {
            return null
        }

    }

    fun setObjectStatus(objStat: TrackStatus){
        this.objStatus  = objStat
    }

    fun getObjectStatus(): TrackStatus{
        return this.objStatus
    }


}

enum class TrackStatus {
    NEW,
    MATURE,
    COASTED
}





