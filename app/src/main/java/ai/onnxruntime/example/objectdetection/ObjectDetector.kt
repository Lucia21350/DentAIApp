package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.InputStream
import java.nio.ByteBuffer
import java.util.*
import android.util.Log


internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>,
    var inferenceTimeMs: Double // 추론 시간 추가
) {}

internal class ObjectDetector(
) {

    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        // Step 1: Convert image into byte array (raw image bytes)
        val rawImageBytes = inputStream.readBytes()

        // Step 2: Get the shape of the byte array and make ort tensor
        val shape = longArrayOf(rawImageBytes.size.toLong())

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            ByteBuffer.wrap(rawImageBytes),
            shape,
            OnnxJavaType.UINT8
        )
        inputTensor.use {
            // Step 3: Call ortSession.run
            val startTime = System.nanoTime()

            val output = ortSession.run(
                mapOf("image" to inputTensor),
                setOf("image_out") // 모델 출력 이름 사용
            )

            val endTime = System.nanoTime()
            val inferenceTimeMs = (endTime - startTime) / 1_000_000.0 // 밀리초로 변환

            Log.d("InferenceTime", "Inference time: $inferenceTimeMs ms")


            // Step 4: Output analysis
            output.use {
                val rawOutput = (output?.get(0)?.value) as ByteArray
                val outputImageBitmap = byteArrayToBitmap(rawOutput)

                // Step 5: Set output result (box output 제거)
                return Result(outputImageBitmap, emptyArray(), inferenceTimeMs) // 박스 정보가 없으면 빈 배열 반환
            }
        }
    }


    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }
}