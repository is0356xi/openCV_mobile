package com.example.opencv_mobile

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.net.Uri
import android.widget.Toast
import com.labters.documentscanner.ImageCropActivity
import com.labters.documentscanner.helpers.ScannerConstants
import kotlinx.android.synthetic.main.activity_img.*


class imgActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_img)

        val uri = intent.getParcelableExtra("uri") as Uri?

        val bmp: Bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)


        var imageView = findViewById<ImageView>(R.id.capturedImg)
        imageView.setImageBitmap(bmp)

        ngBtn.setOnClickListener{
            manual_crop(bmp)
        }
        okBtn.setOnClickListener{
            finish()
        }
    }
    private fun manual_crop(bmp: Bitmap){
        ScannerConstants.selectedImageBitmap=bmp
////        startActivityForResult(Intent(MainActivity@this, ImageCropActivity::class.java), Constants.REQUEST_CROP)
        startActivityForResult(Intent(imgActivity@this, ImageCropActivity::class.java), 1234)

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 1234 && resultCode == Activity.RESULT_OK) {
            if (ScannerConstants.selectedImageBitmap != null) {
//
//                var bmp : Bitmap? = getcontour(ScannerConstants.selectedImageBitmap)
//
//                if(bmp==null){
//                    Toast.makeText(MainActivity@this,"書類検出不可", Toast.LENGTH_LONG).show()
//                }else{
//                    val uri: Uri = bitmapToUri(bmp)
//                    intent_for_CVimg(uri)
//                }

                var imageView = findViewById<ImageView>(R.id.capturedImg)
                imageView.setImageBitmap(ScannerConstants.selectedImageBitmap)

//                val uri: Uri = bitmapToUri(ScannerConstants.selectedImageBitmap)
//                intent_for_CVimg(uri)


            } else
                Toast.makeText(imgActivity@this, "Something wen't wrong.", Toast.LENGTH_LONG)
                    .show()
        }

    }
}
