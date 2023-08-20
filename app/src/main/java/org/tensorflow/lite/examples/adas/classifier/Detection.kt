package org.tensorflow.lite.examples.adas.classifier

import android.graphics.RectF

class Detection {

    private var boundingBox: RectF
    private var category : Category

    constructor(boundingBox: RectF, category: Category){
        this.boundingBox = boundingBox
        this.category = category
    }

    fun getBoundingBox(): RectF{
        return boundingBox
    }

    fun getCategory(): Category{
        return category
    }

    fun getScore(): Float{
        return category.getScore()
    }

    fun getLabel(): String{
        return category.getLabel()
    }

    fun getLabelIndex(): Int{
        return category.getLabelIndex()
    }
}