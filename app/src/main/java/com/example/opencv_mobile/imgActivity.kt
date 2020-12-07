package com.example.opencv_mobile

import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.net.Uri

import kotlinx.android.synthetic.main.activity_img.returnBtn


class imgActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_img)

        val uri = intent.getParcelableExtra("uri") as Uri?

        val bmp: Bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)


        var imageView = findViewById<ImageView>(R.id.capturedImg)
        imageView.setImageBitmap(bmp)

        returnBtn.setOnClickListener{
            finish()
        }
    }

}
