package org.tensorflow.lite.examples.adas.objecttracker

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.MatrixUtils.inverse

class KalmanFilter(dim_x: Int, dim_z: Int) {
    var x: RealVector
    var P: RealMatrix
    var Q: RealMatrix
    var F: RealMatrix
    var H: RealMatrix
    var R: RealMatrix
    private var M: RealMatrix
    private val I: RealMatrix
    private var x_prior: RealVector
    private var P_prior: RealMatrix

    init {
        x = ArrayRealVector(dim_x)
        P = MatrixUtils.createRealIdentityMatrix(dim_x)
        Q = MatrixUtils.createRealIdentityMatrix(dim_x)
        F = MatrixUtils.createRealIdentityMatrix(dim_x)
        H = Array2DRowRealMatrix(dim_z, dim_x)
        R = MatrixUtils.createRealIdentityMatrix(dim_z)
        M = Array2DRowRealMatrix(dim_z, dim_z)

        I = MatrixUtils.createRealIdentityMatrix(dim_x)
        x_prior = x.copy()
        P_prior = P.copy()
    }

    fun predict() {
        x = F.operate(x)
        P = F.multiply(P).multiply(F.transpose()).add(Q)
        x_prior = x.copy()
        P_prior = P.copy()
    }

    fun update(z: RealVector) {
        val y = z.subtract(H.operate(x))
        val PHT = P.multiply(H.transpose())

        val S = H.multiply(PHT).add(R)

        val K = PHT.multiply(inverse(S))

        x = x.add(K.operate(y))
        val I_KH = I.subtract(K.multiply(H))
        P = I_KH.multiply(P)
    }
}
