package com.example.opencv_mobile

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.net.Uri
import android.util.Log
import android.widget.EditText
import android.widget.Toast
import com.google.android.material.textfield.TextInputEditText
import com.labters.documentscanner.ImageCropActivity
import com.labters.documentscanner.helpers.ScannerConstants
import kotlinx.android.synthetic.main.activity_img.*
import org.w3c.dom.Text
import java.io.File
import java.io.FileOutputStream


class imgActivity : AppCompatActivity() {


    private var result_bitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_img)

        // MainからURIを受け取る
        val uri = intent.getParcelableExtra("uri") as Uri?

        // URIからBitmapを取得
        val bmp: Bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)

        // Mainに返却するBitmapの内容を更新
        result_bitmap = bmp


        var imageView = findViewById<ImageView>(R.id.capturedImg)
        imageView.setImageBitmap(bmp)

        val EditText = findViewById<TextInputEditText>(R.id.label)


        ngBtn.setOnClickListener{
            // 正しく書類が認識されていない場合、手動でcropする
            manual_crop(bmp)
        }
        okBtn.setOnClickListener{
            // OKボタンが押されたらcropしたイメージのURIと様式ラベルをMainに返す
            val result = Intent()
            // URIを取得
            var uri: Uri = bitmapToUri(result_bitmap)
            result.putExtra("uri", uri)
            // 様式ラベルを取得
            val label = EditText.getText().toString()
            result.putExtra("label", label)

            setResult(Activity.RESULT_OK, result)
            // 終了
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

                var imageView = findViewById<ImageView>(R.id.capturedImg)
                imageView.setImageBitmap(ScannerConstants.selectedImageBitmap)

                result_bitmap = ScannerConstants.selectedImageBitmap

            } else
                Toast.makeText(imgActivity@this, "Something wen't wrong.", Toast.LENGTH_LONG)
                    .show()
        }
    }

    private fun bitmapToUri(bitmap: Bitmap?): Uri {

        // 一時ファイル作成用のキャッシュディレクトリを定義する
        val cacheDir: File = this.cacheDir

        // 現在日時からファイル名を生成する
        val fileName: String = System.currentTimeMillis().toString() + ".jpg"

        // 空のファイルを生成する
        val file = File(cacheDir, fileName)

        // ファイルにバイトデータを書き込み開始する
        val fileOutputStream: FileOutputStream? = FileOutputStream(file)

        // ファイルにbitmapを書き込む
        bitmap?.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)

        // ファイルにバイトデータを書き込み終了する
        fileOutputStream?.close()

        // ファイルからcontent://スキーマ形式のuriを取得する
//        val contentSchemaUri: Uri = FileProvider.getUriForFile(this, "com.hoge.fuga.fileprovider.fileprovider", file)
        val contentSchemaUri: Uri = Uri.fromFile(file)

        return contentSchemaUri
    }
}
