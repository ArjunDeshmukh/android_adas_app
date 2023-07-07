package org.tensorflow.lite.examples.adas.objecttracker

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealVector

class KalmanTrack(initial_state: DoubleArray) {
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

        kf.P.setSubMatrix(arrayOf(
            doubleArrayOf(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1000.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 10.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 10.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.01)
        ),4, 4) // initial velocity and location error covariance

        kf.Q.setEntry(6, 6, 0.01) // process noise
        kf.Q.setSubMatrix(arrayOf(
            doubleArrayOf(0.01, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.01, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.01, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.01)
        ), 4, 4) // process noise

        kf.x = MatrixUtils.createRealVector(xxyyToXysr(initial_state))   // initialize KalmanFilter state
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
}





