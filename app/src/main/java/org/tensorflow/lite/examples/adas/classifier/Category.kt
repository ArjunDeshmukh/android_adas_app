package org.tensorflow.lite.examples.adas.classifier

class Category(label: String , score: Float) {

    private val label = label
    private val score = score

    fun getLabel(): String{
        return label
    }

    fun getScore(): Float{
        return score
    }

}