package org.tensorflow.lite.examples.adas.objecttracker

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealVector
import org.tensorflow.lite.support.label.Category
class ObjectClass {

    var ID: Int = -1
    var top_left_bottom_right_coord: RealVector = MatrixUtils.createRealVector(doubleArrayOf(-1.0, -1.0, -1.0, -1.0))
    var category: Category? = null
    var filtWidth: Float = -1.0F
    var fTrustObj: Boolean = false

}