/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.adas

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.tensorflow.lite.examples.adas.classifier.TFLiteClassifier
import java.util.ArrayList
import java.util.LinkedList
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results:  MutableList<FloatArray> = mutableListOf()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var inputImageWidth: Int = 640
    private var inputImageHeight: Int = 640
    private var labels = ArrayList<String>()

    private var scaleFactor: Float = 1f

    private var bounds = Rect()

    init {
        initPaints()
        labels = context?.let { TFLiteClassifier.loadLines(it, "coco.txt") }!!
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (result in results) {
            val boundingBox = RectF(
                0f.coerceAtLeast(result[0] - result[2] / 2),
                0f.coerceAtLeast(result[1] - result[3] / 2),
                (inputImageWidth - 1).toFloat()
                    .coerceAtMost(result[0] + result[2] / 2),
                (inputImageHeight - 1).toFloat()
                    .coerceAtMost(result[1] + result[3] / 2)
            )

            val top = 0f.coerceAtLeast(result[1] - result[3] / 2) * scaleFactor
            val bottom = (inputImageHeight - 1).toFloat()
                .coerceAtMost(result[1] + result[3] / 2) * scaleFactor
            val left =  0f.coerceAtLeast(result[0] - result[2] / 2) * scaleFactor
            val right = (inputImageWidth - 1).toFloat()
                .coerceAtMost(result[0] + result[2] / 2) * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText =
                labels[result[6].toInt()] + " " +
                        String.format("%.2f", result[5])

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(
      detectionResults: MutableList<FloatArray>,
      imageHeight: Int,
      imageWidth: Int,
    ) {
        results = detectionResults

        // PreviewView is in FILL_START mode. So we need to scale up the bounding box to match with
        // the size that the captured images will be displayed.
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)

        inputImageWidth = imageWidth
        inputImageHeight = imageHeight

    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
