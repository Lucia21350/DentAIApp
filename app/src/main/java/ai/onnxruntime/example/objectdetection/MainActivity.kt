package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var inputImage: ImageView
    private lateinit var outputImage: ImageView
    private lateinit var objectDetectionButton: Button
    private lateinit var selectImageButton: Button
    private lateinit var classes: List<String>
    private var selectedImageUri: Uri? = null

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImage = findViewById(R.id.imageView1)
        outputImage = findViewById(R.id.imageView2)
        objectDetectionButton = findViewById(R.id.object_detection_button)
        selectImageButton = findViewById(R.id.select_image_button)

        // Initialize classes and ONNX model
        classes = readClasses()
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        // Set up image selection button
        selectImageButton.setOnClickListener {
            openImagePicker()
        }

        // Set up object detection button
        objectDetectionButton.setOnClickListener {
            try {
                selectedImageUri?.let {
                    performObjectDetection(it)
                    Toast.makeText(baseContext, "ObjectDetection performed!", Toast.LENGTH_SHORT).show()
                } ?: run {
                    Toast.makeText(baseContext, "Please select an image first", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when performing ObjectDetection", e)
                Toast.makeText(baseContext, "Failed to perform ObjectDetection", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun updateUI(result: Result) {
        val mutableBitmap: Bitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.color = Color.WHITE
        paint.textSize = 28f
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)
        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, paint)

        result.outputBox.forEach { box ->
            canvas.drawText("%s:%.2f".format(classes[box[5].toInt()], box[4]),
                box[0] - box[2] / 2, box[1] - box[3] / 2, paint)
        }

        outputImage.setImageBitmap(mutableBitmap)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.yolov11n_with_pre_post_processing2
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    // Open image picker
    private fun openImagePicker() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        intent.type = "image/*"
        startActivityForResult(intent, PICK_IMAGE_REQUEST)
    }

    // Handle image picker result
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                selectedImageUri = uri
                inputImage.setImageURI(uri)
            }
        }
    }

    // Perform object detection on selected image
    private fun performObjectDetection(imageUri: Uri) {
        val objDetector = ObjectDetector()

        // Convert the URI to InputStream
        val imageStream: InputStream? = contentResolver.openInputStream(imageUri)

        imageStream?.let {
            inputImage.setImageBitmap(BitmapFactory.decodeStream(it))
            it.reset() // Reset the InputStream before passing to the detector
            val result = objDetector.detect(it, ortEnv, ortSession)
            updateUI(result)
        } ?: run {
            Toast.makeText(baseContext, "Failed to open the selected image", Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        const val TAG = "ORTObjectDetection"
        const val PICK_IMAGE_REQUEST = 1
    }
}
