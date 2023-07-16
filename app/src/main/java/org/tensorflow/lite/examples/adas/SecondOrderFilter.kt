package org.tensorflow.lite.examples.adas

import kotlin.math.PI
import kotlin.math.sqrt
import kotlin.math.tan

data class ButterworthCoefficients(val b0: Float, val b1: Float, val b2: Float, val a0: Float, val a1: Float, val a2: Float)

class SecondOrderFilter(
    fs: Float,
    f0: Float
) {
    private var x1: Float = 0.0F
    private var x2: Float = 0.0F
    private var y1: Float = 0.0F
    private var y2: Float = 0.0F

    private val coefficients = ButterworthCoefficients(b0 = 1.0F, b1 = 2.0F, b2 = 1.0F, a0 = 6.881F, a1 = -4.617F, a2 = 1.736F) // Currently hard coded for fs = 25, f0 = 5

    fun process(input: Float): Float {
        val output = (coefficients.b0 * input + coefficients.b1 * x1 + coefficients.b2 * x2 - coefficients.a1 * y1 - coefficients.a2 * y2)/coefficients.a0

        x2 = x1
        x1 = input
        y2 = y1
        y1 = output

        return output
    }
}