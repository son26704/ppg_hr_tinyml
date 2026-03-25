#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ppg_hr_mlp_int8.h"

namespace
{

    static const char *TAG = "PPG_TINYML";

    constexpr int kInputFeatureSize = 16;
    constexpr float kHrMeanBpm = 96.2f;
    constexpr float kHrStdBpm = 21.7f;

    constexpr int kTensorArenaSize = 16 * 1024;
    alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

    const tflite::Model *g_model = nullptr;
    tflite::MicroInterpreter *g_interpreter = nullptr;
    TfLiteTensor *g_input = nullptr;
    TfLiteTensor *g_output = nullptr;

    static const int8_t kDemoInput[kInputFeatureSize] = {
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
        -40,
    };

    float dequantize_int8(int8_t q, float scale, int zero_point)
    {
        return (static_cast<int>(q) - zero_point) * scale;
    }

    bool tinyml_init()
    {
        g_model = tflite::GetModel(ppg_hr_mlp_int8_tflite);
        if (g_model->version() != TFLITE_SCHEMA_VERSION)
        {
            ESP_LOGE(TAG, "Model schema %d != supported %d", g_model->version(), TFLITE_SCHEMA_VERSION);
            return false;
        }

        static tflite::MicroMutableOpResolver<1> resolver;
        if (resolver.AddFullyConnected() != kTfLiteOk)
        {
            ESP_LOGE(TAG, "AddFullyConnected failed");
            return false;
        }

        static tflite::MicroInterpreter static_interpreter(
            g_model,
            resolver,
            tensor_arena,
            kTensorArenaSize);
        g_interpreter = &static_interpreter;

        if (g_interpreter->AllocateTensors() != kTfLiteOk)
        {
            ESP_LOGE(TAG, "AllocateTensors failed");
            return false;
        }

        g_input = g_interpreter->input(0);
        g_output = g_interpreter->output(0);

        if (g_input == nullptr || g_output == nullptr)
        {
            ESP_LOGE(TAG, "Input/Output tensor is null");
            return false;
        }

        if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8)
        {
            ESP_LOGE(TAG, "Unexpected tensor type. input=%d output=%d", g_input->type, g_output->type);
            return false;
        }

        if (g_input->dims == nullptr || g_input->dims->size != 2 ||
            g_input->dims->data[0] != 1 || g_input->dims->data[1] != kInputFeatureSize)
        {
            ESP_LOGE(TAG, "Unexpected input shape");
            return false;
        }

        ESP_LOGI(TAG, "Model size: %u bytes", ppg_hr_mlp_int8_tflite_len);
        ESP_LOGI(TAG, "Input quant: scale=%.8f, zp=%d", g_input->params.scale, g_input->params.zero_point);
        ESP_LOGI(TAG, "Output quant: scale=%.8f, zp=%d", g_output->params.scale, g_output->params.zero_point);
        ESP_LOGI(TAG, "Tensor arena used: %u / %u bytes",
                 static_cast<unsigned>(g_interpreter->arena_used_bytes()),
                 static_cast<unsigned>(kTensorArenaSize));

        return true;
    }

    void tinyml_run_once(const int8_t *features_q)
    {
        memcpy(g_input->data.int8, features_q, kInputFeatureSize * sizeof(int8_t));

        if (g_interpreter->Invoke() != kTfLiteOk)
        {
            ESP_LOGE(TAG, "Invoke failed");
            return;
        }

        const int8_t y_q = g_output->data.int8[0];
        const float y_norm = dequantize_int8(y_q, g_output->params.scale, g_output->params.zero_point);
        const float y_bpm = y_norm * kHrStdBpm + kHrMeanBpm;

        ESP_LOGI(TAG, "Infer: y_q=%d, y_norm=%.4f, y_bpm=%.2f", static_cast<int>(y_q), y_norm, y_bpm);
    }

} // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting PPG HR TinyML (ESP32-S3)");

    if (!tinyml_init())
    {
        ESP_LOGE(TAG, "TinyML init failed");
        return;
    }

    while (true)
    {
        tinyml_run_once(kDemoInput);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
