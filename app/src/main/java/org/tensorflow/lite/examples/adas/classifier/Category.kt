package org.tensorflow.lite.examples.adas.classifier

import android.content.Context
import android.graphics.RectF
import org.tensorflow.lite.examples.adas.classifier.TFLiteClassifier
import java.io.IOException
import java.io.InputStreamReader
import java.util.ArrayList
import java.util.Scanner

class Category {

    private val label: String
    private val score: Float
    private val labelIndex: Int

    constructor(context: Context, label: String , score: Float)  {
        var labels = loadLines(context, "coco.txt")
        this.label = label
        this.score = score
        this.labelIndex = labels.indexOf(label)
    }

    constructor(context: Context, labelIndex: Int , score: Float)  {
        var labels = loadLines(context, "coco.txt")
        this.score = score
        this.labelIndex = labelIndex
        this.label = labels[this.labelIndex]
    }

    fun getLabel(): String{
        return label
    }

    fun getScore(): Float{
        return score
    }

    fun getLabelIndex(): Int{
        return labelIndex
    }

    @Throws(IOException::class)
    fun loadLines(context: Context, filename: String): ArrayList<String> {
        val s = Scanner(InputStreamReader(context.assets.open(filename)))
        val labels = ArrayList<String>()
        while (s.hasNextLine()) {
            labels.add(s.nextLine())
        }
        s.close()
        return labels
    }

}