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

fun linearSumAssignment(costMatrix: Array<DoubleArray>): Pair<IntArray, IntArray> {
    val numRows = costMatrix.size
    val numCols = costMatrix[0].size

    val linearObjectiveFunction = LinearObjectiveFunction(DoubleArray(numRows * numCols) { i -> costMatrix[i / numCols][i % numCols] }, 0.0)

    val rowConstraints = mutableListOf<LinearConstraint>()
    for (i in 0 until numRows) {
        val coefficients = DoubleArray(numRows * numCols) { j -> if (j / numCols == i) 1.0 else 0.0 }
        rowConstraints.add(LinearConstraint(coefficients, Relationship.EQ, 1.0))
    }

    val colConstraints = mutableListOf<LinearConstraint>()
    for (j in 0 until numCols) {
        val coefficients = DoubleArray(numRows * numCols) { i -> if (i % numCols == j) 1.0 else 0.0 }
        colConstraints.add(LinearConstraint(coefficients, Relationship.EQ, 1.0))
    }

    val constraintSet = LinearConstraintSet(rowConstraints + colConstraints)
    val simplexSolver = SimplexSolver()
    val solution = simplexSolver.optimize(linearObjectiveFunction, constraintSet)

    val assignments = IntArray(numRows)
    for (i in 0 until numRows) {
        val coefficients = solution.point
        val start = i * numCols
        val end = start + numCols
        val assignedCol = coefficients.slice(start until end).maxByOrNull { coefficients[i] }
        assignments[i] = (assignedCol!! % numCols).toInt()
    }

    val unassignedCols = IntArray(numCols) { it }
    val assignedCols = assignments.toHashSet().toIntArray()
    val unassignedRows = IntArray(numRows) { it }
    val assignedRows = IntArray(numRows) { -1 }
    for (i in 0 until numRows) {
        val assignedCol = assignments[i]
        if (assignedCol >= 0) {
            unassignedRows[i] = -1
            unassignedCols[assignedCol] = -1
            assignedRows[i] = assignedCol
        }
    }

    val unassignedRowsIndices = unassignedRows.indices.filter { unassignedRows[it] >= 0 }.toIntArray()
    val unassignedColsIndices = unassignedCols.indices.filter { unassignedCols[it] >= 0 }.toIntArray()

    return assignedRows to assignedCols
}


