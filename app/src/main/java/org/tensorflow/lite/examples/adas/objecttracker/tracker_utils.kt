package org.tensorflow.lite.examples.adas.objecttracker

import org.apache.commons.math3.optim.linear.LinearObjectiveFunction
import org.apache.commons.math3.optim.linear.LinearConstraintSet
import org.apache.commons.math3.optim.linear.LinearConstraint
import org.apache.commons.math3.optim.linear.Relationship
import org.apache.commons.math3.optim.linear.SimplexSolver

fun xysrToXxyy(b: DoubleArray, score: Double? = null): DoubleArray {
    val w = Math.sqrt(b[2] * b[3])
    val h = b[2] / w
    return if (score == null) {
        doubleArrayOf(b[0] - w / 2.0, b[1] - h / 2.0, b[0] + w / 2.0, b[1] + h / 2.0)
    } else {
        doubleArrayOf(b[0] - w / 2.0, b[1] - h / 2.0, b[0] + w / 2.0, b[1] + h / 2.0, score)
    }
}

fun xxyyToXysr(bbox: DoubleArray): DoubleArray {
    val w = bbox[2] - bbox[0]
    val h = bbox[3] - bbox[1]
    val x = bbox[0] + w / 2.0
    val y = bbox[1] + h / 2.0
    val s = w * h
    val r = w / h
    return doubleArrayOf(x, y, s, r)
}

fun bboxToCentroid(bbox: DoubleArray): Pair<Int, Int> {
    val cX = ((bbox[0] + bbox[2]) / 2.0).toInt()
    val cY = ((bbox[1] + bbox[3]) / 2.0).toInt()
    return cX to cY
}

fun greedyAssignment(costMatrix: Array<DoubleArray>): MutableList<Pair<Int, Int>> {
    val numRows = costMatrix.size
    val numCols = costMatrix[0].size

    val assignedRowsCols: MutableList<Pair<Int, Int>> = mutableListOf()
    val usedCols = BooleanArray(numCols)

    for (row in 0 until numRows) {
        var minCost = Double.POSITIVE_INFINITY
        var minCol = -1

        for (col in 0 until numCols) {
            if (!usedCols[col] && costMatrix[row][col] < minCost) {
                minCost = costMatrix[row][col]
                minCol = col
            }
        }

        if (minCol != -1) {
            usedCols[minCol] = true
            assignedRowsCols.add(row to minCol)
        }
    }

    return assignedRowsCols
}






