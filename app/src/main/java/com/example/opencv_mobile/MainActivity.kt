package com.example.opencv_mobile

// OpenCV用
// camerax用

// camscanner用
import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.squareup.moshi.Moshi
import kotlinx.android.synthetic.main.activity_main.*
import okhttp3.MediaType
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import retrofit2.Retrofit
import retrofit2.adapter.rxjava.RxJavaCallAdapterFactory
import retrofit2.converter.moshi.MoshiConverterFactory
import rx.Subscriber
import rx.schedulers.Schedulers
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.pow
import kotlin.math.roundToInt


class MainActivity : AppCompatActivity() {
    private var imageCapture: ImageCapture? = null

//    private var imgData: Bitmap? = null

    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // OpenCVのロード
        if(!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "error_openCV")
        }

        // カメラの権限チェックをして、権限があればカメラ起動処理を呼び出す
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // 撮影ボタンにリスナーを追加する
        camera_capture_button.setOnClickListener { takePhoto() }
        // 解析開始ボタンにリスナーを追加する
//        inferBtn.setOnClickListener { capture_analyze() }

        // 写真を吐き出すディレクトリの取得
        outputDirectory = getOutputDirectory()
        // カメラを別スレッドで動かす
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    // 写真撮影用の関数
    private fun takePhoto() {
        Log.e(TAG, "takePhoto!!!!!!")
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time-stamped output file to hold the image
        val photoFile = File(
            outputDirectory,
            SimpleDateFormat(FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis()) + ".jpg")

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()


        var imgData: Bitmap

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }


                override fun onImageSaved(output: ImageCapture.OutputFileResults) {

                    val inputStream = FileInputStream(photoFile)
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                    val bitmapWidth = bitmap.width
                    val bitmapHeight = bitmap.height
                    val matrix = Matrix()
                    matrix.setRotate(0F, bitmapWidth / 2F, bitmapHeight / 2F)
                    matrix.setRotate(0F, bitmapWidth / 2F, bitmapHeight / 2F)
                    val rotatedBitmap = Bitmap.createBitmap(
                        bitmap,
                        0,
                        0,
                        bitmapWidth,
                        bitmapHeight,
                        matrix,
                        true
                    )

                    imgData = rotatedBitmap



                    var bmp : Bitmap? = getcontour(imgData)

                    var uri: Uri = bitmapToUri(bmp)

                    intent_for_CVimg(uri)

                    val msg = "Photo capture succeeded: ${photoFile.absolutePath}"
                    viewFinder.post {
                        Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    }

                }
            })



    }

    private fun startCamera() {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            Log.e(TAG, "call startcamera!!!!!!")
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.createSurfaceProvider())
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            imageCapture = ImageCapture.Builder()
                .build()

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture)

                Log.e(TAG, "bind lificycle!!!!!")

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))

        Log.e(TAG, "end StartCamera!!!!!")
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    // 2値化処理
    private fun cv_thresh(bmp: Bitmap): Bitmap {
        // bitmapの整形(今回はいらないかも)
        val bmp = Bitmap.createScaledBitmap(bmp, bmp.width, bmp.height, true)
        // Matオブジェクトに変換
        var mat = Mat()
        Utils.bitmapToMat(bmp, mat)

        // グレースケール --> 2値化
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY)  // まずグレースケールへ(明るさだけの形式)
        Imgproc.threshold(mat, mat, 150.0, 255.0, Imgproc.THRESH_OTSU)  // 明るさが0を境に白と黒へ変換


        // 2値化したMatをbitmapに変換
        var th_bmp = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, th_bmp)

        return th_bmp
    }

    // 輪郭抽出
    private fun getcontour(bmp: Bitmap): Bitmap?{

        // Matオブジェクトに変換
        var org_mat = Mat()
        Utils.bitmapToMat(bmp, org_mat)

        // 2値化処理を呼び出してMatに変換
        val th_bmp = cv_thresh(bmp)
        var th_mat = Mat()
        Utils.bitmapToMat(th_bmp, th_mat)

        // 輪郭抽出処理
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.cvtColor(th_mat, th_mat, Imgproc.COLOR_RGB2GRAY)
        Imgproc.findContours(th_mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1)

        // 最大の面積のものを選択
        val max = MatOfPoint2f()
        contours.maxBy { Imgproc.contourArea(it) }?.convertTo(max, CvType.CV_32F)


        // 輪郭を凸形で近似
        val epsilon = 0.1 * Imgproc.arcLength(max, true)
        val approx = MatOfPoint2f()
        Imgproc.approxPolyDP(max, approx, epsilon, true)

        if (approx.rows() != 4) {
            // 角が4つじゃない場合（四角形でない場合）は検出失敗として、そのまま画像を返す
            Toast.makeText(applicationContext, "書類を検出できませんでした", Toast.LENGTH_SHORT).show()

//            return null
            return bmp
        }



        // 書類の横幅
        val cardImageLongSide = 2400.0
        val cardImageShortSide = (cardImageLongSide * (2.4 / 3.56)).roundToInt().toDouble()

        val line1Len = (approx[1, 0][0] - approx[0, 0][0]).pow(2) + (approx[1, 0][1] - approx[0, 0][1]).pow(2)
        val line2Len = (approx[3, 0][0] - approx[2, 0][0]).pow(2) + (approx[3, 0][1] - approx[2, 0][1]).pow(2)
        val line3Len = (approx[2, 0][0] - approx[1, 0][0]).pow(2) + (approx[2, 0][1] - approx[1, 0][1]).pow(2)
        val line4Len = (approx[0, 0][0] - approx[3, 0][0]).pow(2) + (approx[0, 0][1] - approx[3, 0][1]).pow(2)
        val targetLine1 = if (line1Len > line2Len) line1Len else line2Len
        val targetLine2 = if (line3Len > line4Len) line3Len else line4Len

        val cardImageWidth: Double
        val cardImageHeight: Double
        if (targetLine1 > targetLine2) {
            // 縦長
            cardImageWidth = cardImageShortSide
            cardImageHeight = cardImageLongSide
        } else {
            // 横長
            cardImageWidth = cardImageLongSide
            cardImageHeight = cardImageShortSide

        }

        val src = Mat(4, 2, CvType.CV_32F)
        for (i in 0 until 4) {
            src.put(i, 0, *(approx.get(i, 0)))
        }
        val dst = Mat(4, 2, CvType.CV_32F)
        dst.put(0, 0, 0.0, 0.0)
        dst.put(1, 0, 0.0, cardImageHeight)
        dst.put(2, 0, cardImageWidth, cardImageHeight)
        dst.put(3, 0, cardImageWidth, 0.0)

        val projectMatrix = Imgproc.getPerspectiveTransform(src, dst)

        var transformed = Mat()
        Utils.bitmapToMat(bmp, transformed)
        Imgproc.warpPerspective(org_mat, transformed, projectMatrix, Size(cardImageWidth, cardImageHeight))

        // 横長の場合は90度回転させる
        if (cardImageHeight < cardImageWidth) {
            Core.rotate(transformed, transformed, Core.ROTATE_90_CLOCKWISE)
        }

//        Core.rotate(transformed, transformed, Core.ROTATE_90_CLOCKWISE)

        val newBitmap = Bitmap.createBitmap(transformed.width(), transformed.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(transformed, newBitmap)
        return newBitmap

    }


    private fun intent_for_CVimg(uri: Uri){
        var intent = Intent(this, imgActivity::class.java)

        intent.putExtra("uri", uri)

        startActivityForResult(intent, 1)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode != 1) { return }

        if (resultCode == Activity.RESULT_OK && data != null) {
            // IntentからURIを取得
            val uri = data.getParcelableExtra("uri") as Uri?
            var label = data.getStringExtra("label") as String

            // URIをBitmapとして取得
            var bmp: Bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)

            // "様式xx" の部分のみのBitmapを取得
//            bmp = getlabel(bmp)

            // BitmapをFileに変換し、サーバにPOSTする
            val file = bitmapTofile(bmp)
            Log.e("label", label)
            img_post(1, label, file)

        } else if(resultCode == Activity.RESULT_CANCELED) {
            Log.e("ImgCropIntent", "キャンセルされたよ !!!!!!")
        }

    }

    // 様式xxの部分のBitmapを取得する関数
    private  fun getlabel(bitmap: Bitmap?): Bitmap{
        //幅と高さの取得
        //幅と高さの取得
        val imageWidth: Int = bitmap!!.getWidth()
        val imageHeight: Int = bitmap!!.getHeight()


        //トリミングする幅、高さ、座標の設定
        val nWidth = (imageWidth * 0.20f).toInt()
        val nHeight = (imageHeight * 0.1).toInt()
        val startX = (imageWidth * 0.07f).toInt()
        val startY = 0
//        val startY = (imageHeight - nHeight) / 2


        //トリミングしたBitmapの作成
        val newbitmap = Bitmap.createBitmap(
                    bitmap,
                    startX,
                    startY,
                    nWidth,
                    nHeight,
                    null,
                    true
                )

        return newbitmap
    }

    private fun bitmapTofile(bitmap: Bitmap?): File {

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

        return file
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

    private fun img_post(userId: Int, body: String, image: File) {
        Log.e("imgActivity", "img POST !!!!!!!")
        // =====送信するデータを整形する=====
        val map: MutableMap<String, RequestBody> = HashMap() // このオブジェクトを送信する

        val userId = RequestBody.create(MediaType.parse("text/plain"), userId.toString())
        val body = RequestBody.create(MediaType.parse("text/plain"), body)


        val requestBody = RequestBody.create(MediaType.parse("image/jpg"), image) // 拡張子は任意で変えてください
        map.put("user_id", userId)
        map.put("body", body)
        map.put("img_file\"; filename=\"image.png\"", requestBody)

        Log.e("img_post", body.toString())

        createApiClient().ImgPostEntry(params = map)
            .subscribeOn(Schedulers.newThread())
            .observeOn(Schedulers.io())
            .subscribe(object: Subscriber<PostResponse>() {
                override fun onNext(r: PostResponse?) {  // 成功
                    Log.e("imgActivity", r.toString())
                }
                override fun onError(e: Throwable?) {
                    Log.e("imgActivity", e.toString())  // 失敗
                }
                override fun onCompleted() {
                }
            })
    }
    /**
     * APIクライアントのインスタンスを生成して返却する関数
     */

    private fun createApiClient(): ApiInterface {
        val moshi = Moshi.Builder()
            .build()
        val okClient = OkHttpClient()

        val builder = Retrofit.Builder()
            .client(okClient)
            .addCallAdapterFactory(RxJavaCallAdapterFactory.create())
            .addConverterFactory(MoshiConverterFactory.create(moshi))
            .baseUrl("http://192.168.100.60:5000/")
            .build()
        return builder.create(ApiInterface::class.java)
    }

    companion object {
        private const val TAG = "CameraXBasic"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
