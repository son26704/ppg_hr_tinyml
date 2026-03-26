#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/i2c.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ppg_hr_mlp_int8.h"

namespace
{
    static const char *TAG = "PPG_TINYML";

    constexpr int kFeatureCount = 16;
    constexpr float kModelFs = 64.0f;
    constexpr int kWindowSec = 8;
    constexpr int kStrideSec = 2;
    constexpr int kWindowSamplesModel = static_cast<int>(kModelFs) * kWindowSec;

    constexpr int kSensorFs = 100;
    constexpr int kWindowSamplesSensor = kSensorFs * kWindowSec;
    constexpr int kStrideSamplesSensor = kSensorFs * kStrideSec;

    constexpr float kHrMeanBpm = 89.3519745f;
    constexpr float kHrStdBpm = 22.6059856f;

    constexpr int kTensorArenaSize = 16 * 1024;
    alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

    const tflite::Model *g_model = nullptr;
    tflite::MicroInterpreter *g_interpreter = nullptr;
    TfLiteTensor *g_input = nullptr;
    TfLiteTensor *g_output = nullptr;

    constexpr i2c_port_t I2C_PORT = I2C_NUM_0;
    constexpr gpio_num_t I2C_SDA = GPIO_NUM_8;
    constexpr gpio_num_t I2C_SCL = GPIO_NUM_9;
    constexpr uint32_t I2C_FREQ_HZ = 100000;
    constexpr uint8_t MAX30102_ADDR = 0x57;

    constexpr uint8_t REG_FIFO_WR_PTR = 0x04;
    constexpr uint8_t REG_OVF_COUNTER = 0x05;
    constexpr uint8_t REG_FIFO_RD_PTR = 0x06;
    constexpr uint8_t REG_FIFO_DATA = 0x07;
    constexpr uint8_t REG_FIFO_CONFIG = 0x08;
    constexpr uint8_t REG_MODE_CONFIG = 0x09;
    constexpr uint8_t REG_SPO2_CONFIG = 0x0A;
    constexpr uint8_t REG_LED1_PA = 0x0C;
    constexpr uint8_t REG_LED2_PA = 0x0D;
    constexpr uint8_t REG_PART_ID = 0xFF;

    static float g_ir_ring[kWindowSamplesSensor] = {0};
    static int g_ir_head = 0;
    static int g_ir_count = 0;
    static int g_since_last_infer = 0;
    static float g_bpm_ema = 0.0f;
    static bool g_has_bpm_ema = false;
    static TaskHandle_t inference_task_handle = nullptr;
    static portMUX_TYPE g_ring_mux = portMUX_INITIALIZER_UNLOCKED;

    static float g_win_sensor[kWindowSamplesSensor] = {0};
    static float g_win_model[kWindowSamplesModel] = {0};
    static float g_feat[kFeatureCount] = {0};

    static float g_scratch_x[kWindowSamplesModel] = {0};
    static float g_scratch_tmp[kWindowSamplesModel] = {0};
    static float g_scratch_dx[kWindowSamplesModel - 1] = {0};
    static float g_scratch_ac[kWindowSamplesModel] = {0};
    static int g_scratch_peaks[128] = {0};
    static float g_scratch_proms[128] = {0};
    static float g_scratch_hr_inst[127] = {0};
    static float g_scratch_pxx[129] = {0};

    static const float kScalerMean[kFeatureCount] = {
        -0.0737763546f, 1.16286546f, 6.71091704f, 1.17152003f,
        0.874481609f, 0.142864371f, 12.2170699f, 1.52713374f,
        98.2353491f, 23.575453f, 2.6134389f, 0.479753285f,
        67.7734996f, 0.907376801f, 0.411012795f, 79.7886578f,
    };

    static const float kScalerScale[kFeatureCount] = {
        0.116390851f, 0.450887383f, 4.66606231f, 0.449539888f,
        0.192789786f, 0.0380347511f, 2.29328924f, 0.286661155f,
        18.9886333f, 11.6544007f, 0.630249684f, 0.210682037f,
        16.7382242f, 0.0549132159f, 0.0611779285f, 22.5167811f,
    };

    inline float clampf(float v, float lo, float hi)
    {
        if (v < lo)
            return lo;
        if (v > hi)
            return hi;
        return v;
    }

    int cmp_float(const void *a, const void *b)
    {
        const float fa = *static_cast<const float *>(a);
        const float fb = *static_cast<const float *>(b);
        return (fa > fb) - (fa < fb);
    }

    float median_of(float *x, int n)
    {
        if (n <= 0)
            return 0.0f;
        qsort(x, static_cast<size_t>(n), sizeof(float), cmp_float);
        if (n & 1)
            return x[n / 2];
        return 0.5f * (x[n / 2 - 1] + x[n / 2]);
    }

    float mean_of(const float *x, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += x[i];
        return (n > 0) ? sum / static_cast<float>(n) : 0.0f;
    }

    float std_of(const float *x, int n, float mean)
    {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float d = x[i] - mean;
            acc += d * d;
        }
        return (n > 0) ? sqrtf(acc / static_cast<float>(n)) : 0.0f;
    }

    void detrend_linear(float *x, int n)
    {
        if (n < 3)
            return;
        const float n_f = static_cast<float>(n);
        const float sum_t = (n_f - 1.0f) * n_f * 0.5f;
        const float sum_t2 = (n_f - 1.0f) * n_f * (2.0f * n_f - 1.0f) / 6.0f;

        float sum_x = 0.0f;
        float sum_tx = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            sum_x += x[i];
            sum_tx += static_cast<float>(i) * x[i];
        }

        const float den = n_f * sum_t2 - sum_t * sum_t;
        if (fabsf(den) < 1e-9f)
            return;

        const float slope = (n_f * sum_tx - sum_t * sum_x) / den;
        const float intercept = (sum_x - slope * sum_t) / n_f;

        for (int i = 0; i < n; ++i)
        {
            x[i] -= slope * static_cast<float>(i) + intercept;
        }
    }

    void simple_bandpass(float *x, int n, float fs)
    {
        const float dt = 1.0f / fs;
        const float hp_tau = 1.0f / (2.0f * 3.14159265f * 0.7f);
        const float lp_tau = 1.0f / (2.0f * 3.14159265f * 5.0f);
        const float hp_alpha = hp_tau / (hp_tau + dt);
        const float lp_alpha = dt / (lp_tau + dt);

        float hp_prev = 0.0f;
        float x_prev = x[0];
        for (int i = 0; i < n; ++i)
        {
            const float hp = hp_alpha * (hp_prev + x[i] - x_prev);
            hp_prev = hp;
            x_prev = x[i];
            x[i] = hp;
        }

        float lp_prev = x[0];
        for (int i = 0; i < n; ++i)
        {
            lp_prev = lp_prev + lp_alpha * (x[i] - lp_prev);
            x[i] = lp_prev;
        }
    }

    void robust_zscore(float *x, int n)
    {
        memcpy(g_scratch_tmp, x, sizeof(float) * static_cast<size_t>(n));
        const float med = median_of(g_scratch_tmp, n);
        for (int i = 0; i < n; ++i)
            g_scratch_tmp[i] = fabsf(x[i] - med);
        float mad = median_of(g_scratch_tmp, n);
        float scale = 1.4826f * mad;
        if (scale < 1e-8f)
            scale = std_of(x, n, mean_of(x, n));
        if (scale < 1e-8f)
            scale = 1.0f;
        for (int i = 0; i < n; ++i)
            x[i] = (x[i] - med) / scale;
    }

    void normalized_autocorr(const float *x, int n, float *ac)
    {
        float mean = mean_of(x, n);
        float denom = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float d = x[i] - mean;
            denom += d * d;
        }
        if (denom < 1e-8f)
        {
            for (int i = 0; i < n; ++i)
                ac[i] = 0.0f;
            return;
        }
        for (int lag = 0; lag < n; ++lag)
        {
            float num = 0.0f;
            for (int i = 0; i < n - lag; ++i)
            {
                num += (x[i] - mean) * (x[i + lag] - mean);
            }
            ac[lag] = num / denom;
        }
    }

    int find_peaks_simple(const float *x, int n, int min_distance, float prominence_th,
                          int *peaks, float *proms, int max_peaks)
    {
        int count = 0;
        int last = -min_distance;
        for (int i = 1; i < n - 1; ++i)
        {
            if (!(x[i] > x[i - 1] && x[i] >= x[i + 1]))
                continue;
            if (i - last < min_distance)
                continue;

            float left_min = x[i];
            for (int j = i - 1; j >= 0 && j >= i - min_distance; --j)
                if (x[j] < left_min)
                    left_min = x[j];

            float right_min = x[i];
            for (int j = i + 1; j < n && j <= i + min_distance; ++j)
                if (x[j] < right_min)
                    right_min = x[j];

            const float base = (left_min > right_min) ? left_min : right_min;
            const float prom = x[i] - base;
            if (prom < prominence_th)
                continue;

            if (count < max_peaks)
            {
                peaks[count] = i;
                proms[count] = prom;
                ++count;
            }
            last = i;
        }
        return count;
    }

    void resample_linear(const float *in, int n_in, float *out, int n_out)
    {
        if (n_in <= 1 || n_out <= 1)
            return;
        const float scale = static_cast<float>(n_in - 1) / static_cast<float>(n_out - 1);
        for (int i = 0; i < n_out; ++i)
        {
            const float pos = i * scale;
            const int idx = static_cast<int>(pos);
            const float frac = pos - static_cast<float>(idx);
            const int idx2 = (idx + 1 < n_in) ? idx + 1 : idx;
            out[i] = in[idx] * (1.0f - frac) + in[idx2] * frac;
        }
    }

    void compute_psd_features(const float *x, int n, float fs, float &psd_hr_ratio,
                              float &spectral_entropy, float &dom_bpm_hr_band)
    {
        const int nfft = (n >= 256) ? 256 : n;
        if (nfft < 32)
        {
            psd_hr_ratio = 0.0f;
            spectral_entropy = 0.0f;
            dom_bpm_hr_band = 0.0f;
            return;
        }

        const int bins = nfft / 2 + 1;
        const float df = fs / static_cast<float>(nfft);
        float pxx_sum = 0.0f;

        for (int k = 0; k < bins; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;
            for (int t = 0; t < nfft; ++t)
            {
                const float ang = 2.0f * 3.14159265f * static_cast<float>(k * t) / static_cast<float>(nfft);
                re += x[t] * cosf(ang);
                im -= x[t] * sinf(ang);
            }
            const float p = re * re + im * im;
            g_scratch_pxx[k] = (p > 1e-12f) ? p : 1e-12f;
            pxx_sum += g_scratch_pxx[k];
        }

        float total_power = 0.0f;
        float hr_power = 0.0f;
        float best_hr_p = -1.0f;
        float best_hr_f = 0.0f;
        float entropy = 0.0f;

        for (int k = 0; k < bins; ++k)
        {
            const float f = df * static_cast<float>(k);
            const float p = g_scratch_pxx[k];
            if (f >= 0.0f && f <= ((fs / 2.0f < 8.0f) ? fs / 2.0f : 8.0f))
                total_power += p * df;
            if (f >= 0.7f && f <= 3.5f)
            {
                hr_power += p * df;
                if (p > best_hr_p)
                {
                    best_hr_p = p;
                    best_hr_f = f;
                }
            }
            const float prob = p / pxx_sum;
            entropy += -prob * logf(prob);
        }

        psd_hr_ratio = hr_power / (total_power + 1e-8f);
        spectral_entropy = entropy / logf(static_cast<float>(bins));
        dom_bpm_hr_band = (best_hr_p > 0.0f) ? (best_hr_f * 60.0f) : 0.0f;
    }

    void extract_ppg_features(const float *sig_raw, int n, float fs, float *feat)
    {
        memcpy(g_scratch_x, sig_raw, sizeof(float) * static_cast<size_t>(n));

        detrend_linear(g_scratch_x, n);
        simple_bandpass(g_scratch_x, n, fs);
        robust_zscore(g_scratch_x, n);

        float min_v = g_scratch_x[0], max_v = g_scratch_x[0], abs_sum = 0.0f, sq_sum = 0.0f;
        float mean = mean_of(g_scratch_x, n);

        for (int i = 0; i < n; ++i)
        {
            if (g_scratch_x[i] < min_v)
                min_v = g_scratch_x[i];
            if (g_scratch_x[i] > max_v)
                max_v = g_scratch_x[i];
            abs_sum += fabsf(g_scratch_x[i]);
            sq_sum += g_scratch_x[i] * g_scratch_x[i];
            if (i < n - 1)
                g_scratch_dx[i] = g_scratch_x[i + 1] - g_scratch_x[i];
        }
        float std = std_of(g_scratch_x, n, mean);

        const int min_distance = (static_cast<int>(fs * 0.33f) > 1) ? static_cast<int>(fs * 0.33f) : 1;
        const float prominence = ((0.15f * std) > 0.05f) ? (0.15f * std) : 0.05f;

        const int n_peaks = find_peaks_simple(g_scratch_x, n, min_distance, prominence, g_scratch_peaks, g_scratch_proms, 128);

        float hr_est_mean = 0.0f;
        float hr_est_std = 0.0f;
        if (n_peaks >= 2)
        {
            int hr_n = 0;
            for (int i = 0; i < n_peaks - 1; ++i)
            {
                const float ibi = static_cast<float>(g_scratch_peaks[i + 1] - g_scratch_peaks[i]) / fs;
                if (ibi > 1e-6f)
                    g_scratch_hr_inst[hr_n++] = 60.0f / ibi;
            }
            if (hr_n > 0)
            {
                hr_est_mean = mean_of(g_scratch_hr_inst, hr_n);
                hr_est_std = std_of(g_scratch_hr_inst, hr_n, hr_est_mean);
            }
        }

        float peak_prom_mean = 0.0f;
        for (int i = 0; i < n_peaks; ++i)
            peak_prom_mean += g_scratch_proms[i];
        if (n_peaks > 0)
            peak_prom_mean /= static_cast<float>(n_peaks);

        normalized_autocorr(g_scratch_x, n, g_scratch_ac);
        const int lag_min = static_cast<int>(fs * 60.0f / 180.0f);
        int lag_max = static_cast<int>(fs * 60.0f / 40.0f);
        if (lag_max > n - 1)
            lag_max = n - 1;

        float ac_best = 0.0f;
        float ac_best_hr = 0.0f;
        if (lag_max > lag_min)
        {
            int best_lag = lag_min;
            float best_val = g_scratch_ac[lag_min];
            for (int lag = lag_min + 1; lag <= lag_max; ++lag)
            {
                if (g_scratch_ac[lag] > best_val)
                {
                    best_val = g_scratch_ac[lag];
                    best_lag = lag;
                }
            }
            ac_best = best_val;
            ac_best_hr = 60.0f / (static_cast<float>(best_lag) / fs);
        }

        float psd_hr_ratio = 0.0f;
        float spectral_entropy = 0.0f;
        float dom_bpm_hr_band = 0.0f;
        compute_psd_features(g_scratch_x, n, fs, psd_hr_ratio, spectral_entropy, dom_bpm_hr_band);

        float slope_abs_mean = 0.0f;
        for (int i = 0; i < n - 1; ++i)
            slope_abs_mean += fabsf(g_scratch_dx[i]);
        slope_abs_mean = (n > 1) ? (slope_abs_mean / static_cast<float>(n - 1)) : 0.0f;

        feat[0] = mean;
        feat[1] = std;
        feat[2] = max_v - min_v;
        feat[3] = sqrtf(sq_sum / static_cast<float>(n));
        feat[4] = abs_sum / static_cast<float>(n);
        feat[5] = slope_abs_mean;
        feat[6] = static_cast<float>(n_peaks);
        feat[7] = static_cast<float>(n_peaks) / (static_cast<float>(n) / fs);
        feat[8] = hr_est_mean;
        feat[9] = hr_est_std;
        feat[10] = peak_prom_mean;
        feat[11] = ac_best;
        feat[12] = ac_best_hr;
        feat[13] = psd_hr_ratio;
        feat[14] = spectral_entropy;
        feat[15] = dom_bpm_hr_band;

        for (int i = 0; i < kFeatureCount; ++i)
        {
            if (isnan(feat[i]) || isinf(feat[i]))
                feat[i] = 0.0f;
        }
    }

    esp_err_t max30102_write_reg(uint8_t reg, uint8_t value)
    {
        uint8_t data[2] = {reg, value};
        return i2c_master_write_to_device(I2C_PORT, MAX30102_ADDR, data, sizeof(data), pdMS_TO_TICKS(1000));
    }

    esp_err_t max30102_read_reg(uint8_t reg, uint8_t *value)
    {
        return i2c_master_write_read_device(I2C_PORT, MAX30102_ADDR, &reg, 1, value, 1, pdMS_TO_TICKS(1000));
    }

    esp_err_t max30102_read_multi(uint8_t reg, uint8_t *buf, size_t len)
    {
        return i2c_master_write_read_device(I2C_PORT, MAX30102_ADDR, &reg, 1, buf, len, pdMS_TO_TICKS(1000));
    }

    esp_err_t max30102_fifo_pending(uint8_t *pending)
    {
        if (pending == nullptr)
            return ESP_ERR_INVALID_ARG;

        uint8_t wr = 0;
        uint8_t rd = 0;
        uint8_t ovf = 0;
        ESP_RETURN_ON_ERROR(max30102_read_reg(REG_FIFO_WR_PTR, &wr), TAG, "read wr ptr fail");
        ESP_RETURN_ON_ERROR(max30102_read_reg(REG_FIFO_RD_PTR, &rd), TAG, "read rd ptr fail");
        ESP_RETURN_ON_ERROR(max30102_read_reg(REG_OVF_COUNTER, &ovf), TAG, "read ovf fail");

        uint8_t diff = static_cast<uint8_t>((wr - rd) & 0x1F);

        if (ovf > 0)
        {
            ESP_LOGW(TAG, "MAX30102 FIFO overflow=%u, wr=%u rd=%u", static_cast<unsigned>(ovf), static_cast<unsigned>(wr), static_cast<unsigned>(rd));
            if (diff == 0)
            {
                diff = 32;
            }
            max30102_write_reg(REG_OVF_COUNTER, 0x00);
        }

        *pending = diff;
        return ESP_OK;
    }

    esp_err_t max30102_read_sample(uint32_t *red, uint32_t *ir)
    {
        uint8_t raw[6] = {0};
        ESP_RETURN_ON_ERROR(max30102_read_multi(REG_FIFO_DATA, raw, 6), TAG, "read fifo fail");
        uint32_t r = (static_cast<uint32_t>(raw[0]) << 16) | (static_cast<uint32_t>(raw[1]) << 8) | raw[2];
        uint32_t i = (static_cast<uint32_t>(raw[3]) << 16) | (static_cast<uint32_t>(raw[4]) << 8) | raw[5];
        *red = r & 0x03FFFF;
        *ir = i & 0x03FFFF;
        return ESP_OK;
    }

    esp_err_t i2c_init()
    {
        i2c_config_t conf = {};
        conf.mode = I2C_MODE_MASTER;
        conf.sda_io_num = I2C_SDA;
        conf.scl_io_num = I2C_SCL;
        conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
        conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
        conf.master.clk_speed = I2C_FREQ_HZ;

        ESP_RETURN_ON_ERROR(i2c_param_config(I2C_PORT, &conf), TAG, "i2c param config fail");
        return i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0);
    }

    esp_err_t max30102_init()
    {
        uint8_t part_id = 0;
        ESP_RETURN_ON_ERROR(max30102_read_reg(REG_PART_ID, &part_id), TAG, "read part id fail");
        ESP_LOGI(TAG, "MAX30102 PART_ID=0x%02X", part_id);

        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_MODE_CONFIG, 0x40), TAG, "reset fail");
        vTaskDelay(pdMS_TO_TICKS(100));

        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_FIFO_WR_PTR, 0x00), TAG, "fifo wr fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_OVF_COUNTER, 0x00), TAG, "fifo ovf fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_FIFO_RD_PTR, 0x00), TAG, "fifo rd fail");

        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_FIFO_CONFIG, 0x00), TAG, "fifo cfg fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_SPO2_CONFIG, 0x27), TAG, "spo2 cfg fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_LED1_PA, 0x18), TAG, "led1 fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_LED2_PA, 0x18), TAG, "led2 fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_MODE_CONFIG, 0x03), TAG, "mode cfg fail");

        ESP_LOGI(TAG, "MAX30102 init done (100sps)");
        return ESP_OK;
    }

    esp_err_t max30102_recover()
    {
        ESP_LOGW(TAG, "Recovering MAX30102...");
        esp_err_t err = max30102_init();
        if (err == ESP_OK)
        {
            g_ir_head = 0;
            g_ir_count = 0;
            g_since_last_infer = 0;
            g_has_bpm_ema = false;
            ESP_LOGW(TAG, "MAX30102 recover success");
        }
        else
        {
            ESP_LOGE(TAG, "MAX30102 recover failed: %s", esp_err_to_name(err));
        }
        return err;
    }

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

        static tflite::MicroInterpreter static_interpreter(g_model, resolver, tensor_arena, kTensorArenaSize);
        g_interpreter = &static_interpreter;

        if (g_interpreter->AllocateTensors() != kTfLiteOk)
        {
            ESP_LOGE(TAG, "AllocateTensors failed");
            return false;
        }

        g_input = g_interpreter->input(0);
        g_output = g_interpreter->output(0);
        if (g_input == nullptr || g_output == nullptr)
            return false;
        if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8)
            return false;
        if (g_input->dims == nullptr || g_input->dims->size != 2 || g_input->dims->data[1] != kFeatureCount)
            return false;

        ESP_LOGI(TAG, "Model size: %u bytes", ppg_hr_mlp_int8_tflite_len);
        ESP_LOGI(TAG, "Input quant: scale=%.8f, zp=%d", g_input->params.scale, g_input->params.zero_point);
        ESP_LOGI(TAG, "Output quant: scale=%.8f, zp=%d", g_output->params.scale, g_output->params.zero_point);
        ESP_LOGI(TAG, "Tensor arena used: %u / %u bytes",
                 static_cast<unsigned>(g_interpreter->arena_used_bytes()),
                 static_cast<unsigned>(kTensorArenaSize));
        return true;
    }

    void push_ir_sample(float ir)
    {
        bool should_notify = false;

        taskENTER_CRITICAL(&g_ring_mux);
        g_ir_ring[g_ir_head] = ir;
        g_ir_head = (g_ir_head + 1) % kWindowSamplesSensor;
        if (g_ir_count < kWindowSamplesSensor)
            ++g_ir_count;
        ++g_since_last_infer;

        if (g_since_last_infer >= kStrideSamplesSensor && g_ir_count >= kWindowSamplesSensor)
        {
            g_since_last_infer = 0;
            should_notify = true;
        }
        taskEXIT_CRITICAL(&g_ring_mux);

        if (should_notify && inference_task_handle != nullptr)
        {
            xTaskNotifyGive(inference_task_handle);
        }
    }

    bool snapshot_window(float *out)
    {
        taskENTER_CRITICAL(&g_ring_mux);
        if (g_ir_count < kWindowSamplesSensor)
        {
            taskEXIT_CRITICAL(&g_ring_mux);
            return false;
        }
        int idx = g_ir_head;
        for (int i = 0; i < kWindowSamplesSensor; ++i)
        {
            out[i] = g_ir_ring[idx];
            idx = (idx + 1) % kWindowSamplesSensor;
        }
        taskEXIT_CRITICAL(&g_ring_mux);
        return true;
    }

    bool run_inference_once()
    {
        if (!snapshot_window(g_win_sensor))
            return false;

        resample_linear(g_win_sensor, kWindowSamplesSensor, g_win_model, kWindowSamplesModel);

        extract_ppg_features(g_win_model, kWindowSamplesModel, kModelFs, g_feat);

        const float in_scale = g_input->params.scale;
        const int in_zp = g_input->params.zero_point;
        for (int i = 0; i < kFeatureCount; ++i)
        {
            float x_sc = (g_feat[i] - kScalerMean[i]) / (kScalerScale[i] + 1e-8f);
            x_sc = clampf(x_sc, -6.0f, 6.0f);
            float qf = x_sc / in_scale + static_cast<float>(in_zp);
            int qi = static_cast<int>(lrintf(qf));
            if (qi < -128)
                qi = -128;
            if (qi > 127)
                qi = 127;
            g_input->data.int8[i] = static_cast<int8_t>(qi);
        }

        if (g_interpreter->Invoke() != kTfLiteOk)
        {
            ESP_LOGE(TAG, "Invoke failed");
            return false;
        }

        const int8_t y_q = g_output->data.int8[0];
        const float y_norm = dequantize_int8(y_q, g_output->params.scale, g_output->params.zero_point);
        const float y_bpm_raw = y_norm * kHrStdBpm + kHrMeanBpm;

        const float bpm_from_peaks = g_feat[7] * 60.0f;
        const bool quality_ok = (g_feat[11] >= 0.20f) && (bpm_from_peaks >= 40.0f) && (bpm_from_peaks <= 180.0f);

        if (!quality_ok)
        {
            ESP_LOGW(TAG,
                     "Low quality window: ac=%.3f peak_bpm=%.1f std=%.3f",
                     g_feat[11],
                     bpm_from_peaks,
                     g_feat[1]);
            return false;
        }

        const float y_bpm = clampf(y_bpm_raw, 40.0f, 180.0f);
        if (!g_has_bpm_ema)
        {
            g_bpm_ema = y_bpm;
            g_has_bpm_ema = true;
        }
        else
        {
            constexpr float kEmaAlpha = 0.25f;
            g_bpm_ema = (kEmaAlpha * y_bpm) + ((1.0f - kEmaAlpha) * g_bpm_ema);
        }

        ESP_LOGI(TAG,
                 "BPM=%.2f (raw=%.2f) | y_q=%d | std=%.3f peak_rate=%.3f ac=%.3f",
                 g_bpm_ema,
                 y_bpm,
                 static_cast<int>(y_q),
                 g_feat[1],
                 g_feat[7],
                 g_feat[11]);
        return true;
    }

    void inference_task(void *arg)
    {
        while (true)
        {
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
            run_inference_once();
        }
    }
} // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting PPG HR TinyML dual-core (ESP32-S3 + MAX30102)");

    if (!tinyml_init())
    {
        ESP_LOGE(TAG, "TinyML init failed");
        return;
    }

    ESP_ERROR_CHECK(i2c_init());
    ESP_ERROR_CHECK(max30102_init());

    xTaskCreatePinnedToCore(
        inference_task,
        "AI_Task",
        8192,
        nullptr,
        5,
        &inference_task_handle,
        1);

    int64_t last_activity_us = esp_timer_get_time();
    int consecutive_i2c_errors = 0;

    while (true)
    {
        uint8_t pending = 0;
        if (max30102_fifo_pending(&pending) != ESP_OK)
        {
            consecutive_i2c_errors++;
            if (consecutive_i2c_errors >= 5)
            {
                max30102_recover();
                consecutive_i2c_errors = 0;
            }
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }
        consecutive_i2c_errors = 0;

        if (pending == 0)
        {
            int64_t now = esp_timer_get_time();
            if (now - last_activity_us > 3000000)
            {
                ESP_LOGW(TAG, "No new MAX30102 samples for >3s. Sensor might be sleeping/reset!");
                max30102_recover();
                last_activity_us = esp_timer_get_time();
            }
            vTaskDelay(pdMS_TO_TICKS(2));
            continue;
        }

        const uint8_t to_read = pending;
        for (uint8_t n = 0; n < to_read; ++n)
        {
            uint32_t red = 0;
            uint32_t ir = 0;
            if (max30102_read_sample(&red, &ir) == ESP_OK)
            {
                (void)red;
                push_ir_sample(static_cast<float>(ir));
                last_activity_us = esp_timer_get_time();
            }
            else
            {
                consecutive_i2c_errors++;
                if (consecutive_i2c_errors >= 5)
                {
                    max30102_recover();
                    consecutive_i2c_errors = 0;
                }
                break;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
