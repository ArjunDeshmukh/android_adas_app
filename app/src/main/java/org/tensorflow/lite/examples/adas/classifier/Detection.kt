package org.tensorflow.lite.examples.adas.classifier

import android.graphics.RectF

class Detection {

    private lateinit var boundingBox: RectF
    private lateinit var category : Category

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
}