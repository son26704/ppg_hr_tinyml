#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "dsps_fft2r.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ppg_hr_mlp_int8.h"

namespace
{
    static const char *TAG = "PPG_TINYML";

    enum run_mode_t
    {
        RUN_MODE_FIXED_NORMAL = 0,
        RUN_MODE_FIXED_HIGH = 1,
        RUN_MODE_ADAPTIVE = 2,
    };

#ifndef RUN_MODE
#define RUN_MODE RUN_MODE_ADAPTIVE
#endif

    constexpr run_mode_t kRunMode = static_cast<run_mode_t>(RUN_MODE);
    constexpr bool kAdaptiveEnabled = (kRunMode == RUN_MODE_ADAPTIVE);
    constexpr bool kTinyMlEnabledByMode = (kRunMode != RUN_MODE_FIXED_NORMAL);
    constexpr int64_t kSparseLogPeriodUs = 2000LL * 1000LL;
    constexpr gpio_num_t PROFILING_FEATURE_GPIO = GPIO_NUM_10;
    constexpr gpio_num_t PROFILING_INVOKE_GPIO = GPIO_NUM_11;

    constexpr int kFeatureCount = 16;
    constexpr float kModelFs = 64.0f;
    constexpr int kWindowSec = 8;
    constexpr int kStrideSec = 2;
    constexpr int kWindowSamplesModel = static_cast<int>(kModelFs) * kWindowSec;
    constexpr int kPsdFftSize = 256;

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

    struct max30102_profile_t
    {
        uint8_t fifo_config;
        uint8_t spo2_config;
        uint8_t led1_pa;
        uint8_t led2_pa;
        uint8_t mode_config;
        int sample_rate_hz;
        const char *name;
    };

    static const max30102_profile_t kProfile50Med = {
        0x00,
        0x23,
        0x18,
        0x18,
        0x03,
        50,
        "50sps_med",
    };

    static const max30102_profile_t kProfile100Med = {
        0x00,
        0x27,
        0x18,
        0x18,
        0x03,
        100,
        "100sps_med",
    };

    enum scheduler_state_t
    {
        SCHED_STATE_NORMAL = 0,
        SCHED_STATE_HIGH = 1,
    };

    enum quality_fail_reason_t
    {
        QFR_NONE = 0,
        QFR_NO_CONTACT,
        QFR_AMP_FAIL,
        QFR_AC_FAIL,
        QFR_HR_RANGE_FAIL,
        QFR_CONSIST_FAIL,
    };

    struct window_metrics_t
    {
        float peak_bpm = 0.0f;
        float ac_best = 0.0f;
        float ac_best_hr = 0.0f;
        float std_hp = 0.0f;
        float ptp_hp = 0.0f;
    };
    struct profiling_pulse_t
    {
        explicit profiling_pulse_t(gpio_num_t pin) : pin_(pin)
        {
            gpio_set_level(pin_, 1);
        }

        ~profiling_pulse_t()
        {
            gpio_set_level(pin_, 0);
        }

    private:
        gpio_num_t pin_;
    };

    constexpr float SCHED_STD_MIN = 250.0f;
    constexpr float SCHED_PTP_MIN = 1000.0f;
    constexpr float SCHED_PTP_MAX = 35000.0f;
    constexpr float SCHED_AC_MIN = 0.30f;
    constexpr float SCHED_AC_HARD = 0.20f;
    constexpr float SCHED_AC_EASY = 0.40f;
    constexpr float SCHED_DIFF_HARD = 0.80f;
    constexpr float SCHED_DIFF_EASY = 0.60f;
    constexpr int SCHED_BAD_WINDOWS_TO_UP = 4;
    constexpr int SCHED_GOOD_WINDOWS_TO_DOWN = 5;
    constexpr int SCHED_COOLDOWN_WINDOWS = 3;
    constexpr int64_t SCHED_MIN_STATE_DWELL_US = 15LL * 1000LL * 1000LL;
    constexpr float NO_CONTACT_STD_HP_MIN = 6000.0f;
    constexpr float NO_CONTACT_STD_HP_SOFT = 2500.0f;
    constexpr float NO_CONTACT_PTP_HP_MIN = 30000.0f;
    constexpr float NO_CONTACT_PEAK_BPM_MAX = 40.0f;
    constexpr float NO_CONTACT_AC_MAX = 0.20f;
    constexpr float HR_CONSIST_MAX_DIFF_BPM = 18.0f;
    constexpr int DSP_PUBLISH_WINDOWS = 2;
    constexpr int DECISION_FREEZE_ON_ERROR_WINDOWS = 2;
    constexpr int DECISION_FREEZE_ON_RECOVER_WINDOWS = 4;
    constexpr int GRAY_WINDOWS_TO_UP = 4;
    constexpr float AC_HR_CLAMP_LOW = 42.0f;
    constexpr float AC_HR_CLAMP_HIGH = 178.0f;

    static float g_ir_ring[kWindowSamplesSensor] = {0};
    static int g_ir_head = 0;
    static int g_ir_count = 0;
    static int g_since_last_eval = 0;
    static int g_window_samples = 400;
    static int g_stride_samples = 100;
    static int g_current_fs = 50;
    static float g_bpm_ema = 0.0f;
    static bool g_has_bpm_ema = false;
    static TaskHandle_t inference_task_handle = nullptr;
    static portMUX_TYPE g_ring_mux = portMUX_INITIALIZER_UNLOCKED;
    static portMUX_TYPE g_state_mux = portMUX_INITIALIZER_UNLOCKED;
    static scheduler_state_t g_sched_state = SCHED_STATE_NORMAL;
    static scheduler_state_t g_desired_state = SCHED_STATE_NORMAL;
    static bool g_switch_pending = false;
    static int64_t g_last_state_switch_us = 0;
    static int g_decision_freeze_windows = 0;
    static bool g_tinyml_ready = false;

    static portMUX_TYPE g_telemetry_mux = portMUX_INITIALIZER_UNLOCKED;
    static uint32_t g_last_red = 0;
    static uint32_t g_last_ir = 0;
    static float g_last_hr_report = 0.0f;
    static int g_last_hr_source = -1; // 0=dsp, 1=ai
    static int g_last_quality_pass = -1;
    static float g_last_difficulty_proxy = 0.0f;
    static float g_last_ac_best = 0.0f;
    static float g_last_peak_bpm = 0.0f;

                
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
    static float g_scratch_pxx[(kPsdFftSize / 2) + 1] = {0};
    static float g_scratch_hp[kWindowSamplesSensor] = {0};
    static float g_fft_buf[2 * kPsdFftSize] = {0};
    static bool g_fft_ready = false;

    static const float kScalerMean[kFeatureCount] = {
        -0.0737763546f,
        1.16286546f,
        6.71091704f,
        1.17152003f,
        0.874481609f,
        0.142864371f,
        12.2170699f,
        1.52713374f,
        98.2353491f,
        23.575453f,
        2.6134389f,
        0.479753285f,
        67.7734996f,
        0.907376801f,
        0.411012795f,
        79.7886578f,
    };

    static const float kScalerScale[kFeatureCount] = {
        0.116390851f,
        0.450887383f,
        4.66606231f,
        0.449539888f,
        0.192789786f,
        0.0380347511f,
        2.29328924f,
        0.286661155f,
        18.9886333f,
        11.6544007f,
        0.630249684f,
        0.210682037f,
        16.7382242f,
        0.0549132159f,
        0.0611779285f,
        22.5167811f,
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
        const float mean = mean_of(x, n);
        float scale = std_of(x, n, mean);
        if (scale < 1e-6f)
            scale = 1.0f;
        const float inv_scale = 1.0f / scale;
        for (int i = 0; i < n; ++i)
            x[i] = (x[i] - mean) * inv_scale;
    }

    void normalized_autocorr(const float *x, int n, float fs, float bpm_min, float bpm_max, float *ac_best, float *ac_best_hr)
    {
        if (ac_best != nullptr)
            *ac_best = 0.0f;
        if (ac_best_hr != nullptr)
            *ac_best_hr = 0.0f;
        if (x == nullptr || n <= 2 || ac_best == nullptr || ac_best_hr == nullptr)
            return;

        const float mean = mean_of(x, n);
        float denom = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float d = x[i] - mean;
            denom += d * d;
        }
        if (denom < 1e-8f)
            return;

        int lag_min = static_cast<int>(fs * 60.0f / bpm_max);
        int lag_max = static_cast<int>(fs * 60.0f / bpm_min);
        if (lag_min < 1)
            lag_min = 1;
        if (lag_max > n - 1)
            lag_max = n - 1;
        if (lag_max <= lag_min)
            return;

        int best_lag = lag_min;
        float best_val = -1.0f;
        for (int lag = lag_min; lag <= lag_max; ++lag)
        {
            float num = 0.0f;
            for (int i = 0; i < n - lag; ++i)
            {
                num += (x[i] - mean) * (x[i + lag] - mean);
            }
            const float ac = num / denom;
            if (ac > best_val)
            {
                best_val = ac;
                best_lag = lag;
            }
        }

        if (best_val < 0.0f)
            best_val = 0.0f;
        *ac_best = best_val;
        *ac_best_hr = 60.0f / (static_cast<float>(best_lag) / fs);
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

    bool ensure_fft_ready()
    {
        if (g_fft_ready)
            return true;
        const esp_err_t err = dsps_fft2r_init_fc32(nullptr, kPsdFftSize);
        if (err != ESP_OK)
        {
            ESP_LOGE(TAG, "dsps_fft2r_init_fc32 failed: %s", esp_err_to_name(err));
            return false;
        }
        g_fft_ready = true;
        return true;
    }

    void compute_psd_features(const float *x, int n, float fs, float &psd_hr_ratio,
                              float &spectral_entropy, float &dom_bpm_hr_band)
    {
        psd_hr_ratio = 0.0f;
        spectral_entropy = 0.0f;
        dom_bpm_hr_band = 0.0f;

        if (x == nullptr || n < kPsdFftSize || !ensure_fft_ready())
            return;

        const int nfft = kPsdFftSize;
        const int bins = nfft / 2 + 1;
        const float df = fs / static_cast<float>(nfft);

        for (int i = 0; i < nfft; ++i)
        {
            g_fft_buf[2 * i + 0] = x[i];
            g_fft_buf[2 * i + 1] = 0.0f;
        }

        dsps_fft2r_fc32(g_fft_buf, nfft);
        dsps_bit_rev_fc32(g_fft_buf, nfft);

        float pxx_sum = 0.0f;
        for (int k = 0; k < bins; ++k)
        {
            const float re = g_fft_buf[2 * k + 0];
            const float im = g_fft_buf[2 * k + 1];
            const float p = re * re + im * im;
            g_scratch_pxx[k] = (p > 1e-12f) ? p : 1e-12f;
            pxx_sum += g_scratch_pxx[k];
        }
        if (pxx_sum < 1e-12f)
            return;

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

        // Use stricter peak constraints to reduce double-peak overcount on fingertip signals.
        const int min_distance = (static_cast<int>(fs * 60.0f / 140.0f) > 1) ? static_cast<int>(fs * 60.0f / 140.0f) : 1;
        const float prominence = ((0.25f * std) > 0.12f) ? (0.25f * std) : 0.12f;

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

        float ac_best = 0.0f;
        float ac_best_hr = 0.0f;
        normalized_autocorr(g_scratch_x, n, fs, 40.0f, 180.0f, &ac_best, &ac_best_hr);

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
    int profile_id_from_state(scheduler_state_t state)
    {
        return (state == SCHED_STATE_HIGH) ? 0 : 1;
    }

    void print_sparse_csv_log(int64_t timestamp_ms)
    {
        scheduler_state_t state_snapshot;
        taskENTER_CRITICAL(&g_state_mux);
        state_snapshot = g_sched_state;
        taskEXIT_CRITICAL(&g_state_mux);

        uint32_t red_snapshot = 0;
        uint32_t ir_snapshot = 0;
        int quality_snapshot = -1;
        float diff_snapshot = 0.0f;

        taskENTER_CRITICAL(&g_telemetry_mux);
        red_snapshot = g_last_red;
        ir_snapshot = g_last_ir;
        quality_snapshot = g_last_quality_pass;
        diff_snapshot = g_last_difficulty_proxy;
        taskEXIT_CRITICAL(&g_telemetry_mux);

        printf("%lld,%d,%d,%d,%.3f,%u,%u\n",
               static_cast<long long>(timestamp_ms),
               static_cast<int>(state_snapshot),
               profile_id_from_state(state_snapshot),
               quality_snapshot,
               diff_snapshot,
               static_cast<unsigned>(red_snapshot),
               static_cast<unsigned>(ir_snapshot));
    }

    esp_err_t max30102_apply_profile(const max30102_profile_t *profile)
    {
        if (profile == nullptr)
            return ESP_ERR_INVALID_ARG;

        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_MODE_CONFIG, 0x40), TAG, "reset fail");
        vTaskDelay(pdMS_TO_TICKS(100));

        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_FIFO_WR_PTR, 0x00), TAG, "fifo wr fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_OVF_COUNTER, 0x00), TAG, "fifo ovf fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_FIFO_RD_PTR, 0x00), TAG, "fifo rd fail");

        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_FIFO_CONFIG, profile->fifo_config), TAG, "fifo cfg fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_SPO2_CONFIG, profile->spo2_config), TAG, "spo2 cfg fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_LED1_PA, profile->led1_pa), TAG, "led1 fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_LED2_PA, profile->led2_pa), TAG, "led2 fail");
        ESP_RETURN_ON_ERROR(max30102_write_reg(REG_MODE_CONFIG, profile->mode_config), TAG, "mode cfg fail");

        ESP_LOGI(TAG, "MAX30102 profile applied: %s (%dsps)", profile->name, profile->sample_rate_hz);
        return ESP_OK;
    }

    void add_decision_freeze_windows(int n)
    {
        if (n <= 0)
            return;

        taskENTER_CRITICAL(&g_state_mux);
        if (g_decision_freeze_windows < n)
        {
            g_decision_freeze_windows = n;
        }
        taskEXIT_CRITICAL(&g_state_mux);
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
        ESP_LOGI(TAG, "MAX30102 probe done");
        return ESP_OK;
    }

    void update_sampling_params_locked(scheduler_state_t state)
    {
        if (state == SCHED_STATE_HIGH)
        {
            g_current_fs = 100;
            g_window_samples = 100 * kWindowSec;
            g_stride_samples = 100 * kStrideSec;
        }
        else
        {
            g_current_fs = 50;
            g_window_samples = 50 * kWindowSec;
            g_stride_samples = 50 * kStrideSec;
        }
    }

    esp_err_t apply_scheduler_state(scheduler_state_t state)
    {
        const max30102_profile_t *profile = (state == SCHED_STATE_HIGH) ? &kProfile100Med : &kProfile50Med;
        ESP_RETURN_ON_ERROR(max30102_apply_profile(profile), TAG, "apply profile fail");

        taskENTER_CRITICAL(&g_ring_mux);
        g_ir_head = 0;
        g_ir_count = 0;
        g_since_last_eval = 0;
        taskEXIT_CRITICAL(&g_ring_mux);

        taskENTER_CRITICAL(&g_state_mux);
        g_sched_state = state;
        update_sampling_params_locked(state);
        g_last_state_switch_us = esp_timer_get_time();
        taskEXIT_CRITICAL(&g_state_mux);

        ESP_LOGI(TAG, "Scheduler state -> %s", (state == SCHED_STATE_HIGH) ? "HIGH(100sps)" : "NORMAL(50sps)");
        return ESP_OK;
    }

    esp_err_t max30102_recover()
    {
        ESP_LOGW(TAG, "Recovering MAX30102...");
        scheduler_state_t state_snapshot;
        taskENTER_CRITICAL(&g_state_mux);
        state_snapshot = g_sched_state;
        taskEXIT_CRITICAL(&g_state_mux);

        esp_err_t err = max30102_init();
        if (err == ESP_OK)
        {
            err = apply_scheduler_state(state_snapshot);
            if (err == ESP_OK)
            {
                g_has_bpm_ema = false;
                add_decision_freeze_windows(DECISION_FREEZE_ON_RECOVER_WINDOWS);
                ESP_LOGW(TAG, "MAX30102 recover success");
            }
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
        ++g_since_last_eval;

        int stride_samples_snapshot;
        int window_samples_snapshot;
        taskENTER_CRITICAL(&g_state_mux);
        stride_samples_snapshot = g_stride_samples;
        window_samples_snapshot = g_window_samples;
        taskEXIT_CRITICAL(&g_state_mux);

        if (g_since_last_eval >= stride_samples_snapshot && g_ir_count >= window_samples_snapshot)
        {
            g_since_last_eval = 0;
            should_notify = true;
        }
        taskEXIT_CRITICAL(&g_ring_mux);

        if (should_notify && inference_task_handle != nullptr)
        {
            xTaskNotifyGive(inference_task_handle);
        }
    }

    bool snapshot_window(float *out, int n_samples)
    {
        taskENTER_CRITICAL(&g_ring_mux);
        if (n_samples <= 0 || n_samples > kWindowSamplesSensor || g_ir_count < n_samples)
        {
            taskEXIT_CRITICAL(&g_ring_mux);
            return false;
        }
        int idx = (g_ir_head - n_samples + kWindowSamplesSensor) % kWindowSamplesSensor;
        for (int i = 0; i < n_samples; ++i)
        {
            out[i] = g_ir_ring[idx];
            idx = (idx + 1) % kWindowSamplesSensor;
        }
        taskEXIT_CRITICAL(&g_ring_mux);
        return true;
    }

    void compute_hp_metrics(const float *x, int n, float fs_hz, float *std_hp, float *ptp_hp)
    {
        if (x == nullptr || std_hp == nullptr || ptp_hp == nullptr || n <= 1)
        {
            if (std_hp != nullptr)
                *std_hp = 0.0f;
            if (ptp_hp != nullptr)
                *ptp_hp = 0.0f;
            return;
        }

        const float fs = (fs_hz > 1.0f) ? fs_hz : 50.0f;
        const float dt = 1.0f / fs;
        const float tau = 0.8f;
        float alpha = dt / (tau + dt);
        alpha = clampf(alpha, 0.001f, 1.0f);

        float lp = x[0];
        float hp_min = 0.0f;
        float hp_max = 0.0f;
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int i = 0; i < n; ++i)
        {
            lp += alpha * (x[i] - lp);
            const float hp = x[i] - lp;
            g_scratch_hp[i] = hp;
            if (i == 0)
            {
                hp_min = hp;
                hp_max = hp;
            }
            else
            {
                if (hp < hp_min)
                    hp_min = hp;
                if (hp > hp_max)
                    hp_max = hp;
            }
            sum += hp;
            sum_sq += hp * hp;
        }

        const float n_f = static_cast<float>(n);
        const float mean = sum / n_f;
        float var = (sum_sq / n_f) - (mean * mean);
        if (var < 0.0f)
            var = 0.0f;
        *std_hp = sqrtf(var);
        *ptp_hp = hp_max - hp_min;
    }

    bool compute_window_metrics(window_metrics_t *metrics, bool need_full_features)
    {
        if (metrics == nullptr)
            return false;

        int window_samples_snapshot;
        int fs_snapshot;
        taskENTER_CRITICAL(&g_state_mux);
        window_samples_snapshot = g_window_samples;
        fs_snapshot = g_current_fs;
        taskEXIT_CRITICAL(&g_state_mux);

        if (!snapshot_window(g_win_sensor, window_samples_snapshot))
            return false;

        const float fs = static_cast<float>(fs_snapshot);
        compute_hp_metrics(g_win_sensor, window_samples_snapshot, fs, &metrics->std_hp, &metrics->ptp_hp);

        resample_linear(g_win_sensor, window_samples_snapshot, g_win_model, kWindowSamplesModel);

        if (need_full_features)
        {
            profiling_pulse_t feature_pulse(PROFILING_FEATURE_GPIO);
            extract_ppg_features(g_win_model, kWindowSamplesModel, kModelFs, g_feat);
            metrics->peak_bpm = g_feat[7] * 60.0f;
            metrics->ac_best = g_feat[11];
            metrics->ac_best_hr = g_feat[12];
            return true;
        }

        memcpy(g_scratch_x, g_win_model, sizeof(float) * static_cast<size_t>(kWindowSamplesModel));
        
        detrend_linear(g_scratch_x, kWindowSamplesModel); 
        
        simple_bandpass(g_scratch_x, kWindowSamplesModel, kModelFs);
        
        robust_zscore(g_scratch_x, kWindowSamplesModel);

        const float mean = mean_of(g_scratch_x, kWindowSamplesModel);
        const float std = std_of(g_scratch_x, kWindowSamplesModel, mean);
        
        const int min_distance = (static_cast<int>(kModelFs * 60.0f / 140.0f) > 1) ? static_cast<int>(kModelFs * 60.0f / 140.0f) : 1;
        const float prominence = ((0.25f * std) > 0.12f) ? (0.25f * std) : 0.12f;
        
        const int n_peaks = find_peaks_simple(
            g_scratch_x,
            kWindowSamplesModel,
            min_distance,
            prominence,
            g_scratch_peaks,
            g_scratch_proms,
            128);
            
        metrics->peak_bpm = (static_cast<float>(n_peaks) / (static_cast<float>(kWindowSamplesModel) / kModelFs)) * 60.0f;

        metrics->ac_best = 0.0f;
        metrics->ac_best_hr = 0.0f;
        
        normalized_autocorr(g_scratch_x, kWindowSamplesModel, kModelFs, 40.0f, 180.0f, &metrics->ac_best, &metrics->ac_best_hr);
        
        return true;
    }

    bool run_tinyml_on_features(float *y_bpm, int8_t *y_q)
    {
        if (!g_tinyml_ready)
            return false;

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

        {
            profiling_pulse_t invoke_pulse(PROFILING_INVOKE_GPIO);
            const int64_t t_start = esp_timer_get_time();
            if (g_interpreter->Invoke() != kTfLiteOk)
            {
                ESP_LOGE(TAG, "Invoke failed");
                return false;
            }
            const int64_t t_end = esp_timer_get_time();
            ESP_LOGI(TAG, "TinyML Invoke time: %lld us", static_cast<long long>(t_end - t_start));
        }

        const int8_t y_q_local = g_output->data.int8[0];
        const float y_norm = dequantize_int8(y_q_local, g_output->params.scale, g_output->params.zero_point);
        *y_bpm = clampf(y_norm * kHrStdBpm + kHrMeanBpm, 40.0f, 180.0f);
        *y_q = y_q_local;
        return true;
    }

    void inference_task(void *arg)
    {
        int bad_windows = 0;
        int gray_windows = 0;
        int good_windows = 0;
        int dsp_publish_windows = 0;
        int cooldown_windows = 0;

        while (true)
        {
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

            scheduler_state_t state_snapshot;
            int64_t last_switch_snapshot = 0;
            int freeze_windows_snapshot = 0;
            taskENTER_CRITICAL(&g_state_mux);
            state_snapshot = g_sched_state;
            last_switch_snapshot = g_last_state_switch_us;
            freeze_windows_snapshot = g_decision_freeze_windows;
            if (g_decision_freeze_windows > 0)
            {
                g_decision_freeze_windows--;
            }
            taskEXIT_CRITICAL(&g_state_mux);

            window_metrics_t metrics = {};
            const bool need_full_features = (state_snapshot == SCHED_STATE_HIGH);
            if (!compute_window_metrics(&metrics, need_full_features))
                continue;

            const float peak_bpm = metrics.peak_bpm;
            const float ac_best = metrics.ac_best;
            const float std_hp = metrics.std_hp;
            const float ptp_hp = metrics.ptp_hp;
            const float ac_best_hr = metrics.ac_best_hr;

            const bool amplitude_ok = (std_hp >= SCHED_STD_MIN) && (ptp_hp >= SCHED_PTP_MIN) && (ptp_hp <= SCHED_PTP_MAX);
            const bool periodic_ok = (ac_best >= SCHED_AC_MIN);
            const bool hr_range_ok = (peak_bpm >= 40.0f) && (peak_bpm <= 180.0f);
            const bool quality_ok = amplitude_ok && periodic_ok && hr_range_ok;
            const float difficulty_proxy = amplitude_ok ? clampf(1.0f - ac_best, 0.0f, 1.0f) : 1.0f;
            const bool ac_hr_valid = (ac_best_hr >= 40.0f) && (ac_best_hr <= 180.0f) &&
                                     (ac_best_hr > AC_HR_CLAMP_LOW) && (ac_best_hr < AC_HR_CLAMP_HIGH);
            const bool hr_consistent = ac_hr_valid && (fabsf(peak_bpm - ac_best_hr) <= HR_CONSIST_MAX_DIFF_BPM);
            const bool no_contact_hard = (peak_bpm <= NO_CONTACT_PEAK_BPM_MAX) &&
                                         ((std_hp >= NO_CONTACT_STD_HP_MIN) || (ptp_hp >= NO_CONTACT_PTP_HP_MIN));
            const bool no_contact_soft = (peak_bpm <= NO_CONTACT_PEAK_BPM_MAX) &&
                                         (ac_best <= NO_CONTACT_AC_MAX) &&
                                         (std_hp >= NO_CONTACT_STD_HP_SOFT);
            const bool no_contact = no_contact_hard || no_contact_soft;

            const bool amp_fail = !amplitude_ok;
            const bool ac_fail = !periodic_ok;
            const bool range_fail = !hr_range_ok;
            const bool consist_fail = !hr_consistent;

            quality_fail_reason_t fail_reason = QFR_NONE;
            if (no_contact)
            {
                fail_reason = QFR_NO_CONTACT;
            }
            else if (amp_fail)
            {
                fail_reason = QFR_AMP_FAIL;
            }
            else if (range_fail)
            {
                fail_reason = QFR_HR_RANGE_FAIL;
            }
            else if (ac_fail)
            {
                fail_reason = QFR_AC_FAIL;
            }
            else if (consist_fail)
            {
                fail_reason = QFR_CONSIST_FAIL;
            }

            const bool severe_motion_fail = (fail_reason == QFR_AMP_FAIL) || (fail_reason == QFR_HR_RANGE_FAIL);
            const bool mild_fail = (fail_reason == QFR_AC_FAIL) || (fail_reason == QFR_CONSIST_FAIL);

            taskENTER_CRITICAL(&g_telemetry_mux);
            g_last_quality_pass = quality_ok ? 1 : 0;
            g_last_difficulty_proxy = difficulty_proxy;
            g_last_ac_best = ac_best;
            g_last_peak_bpm = peak_bpm;
            taskEXIT_CRITICAL(&g_telemetry_mux);

            const int64_t now_us = esp_timer_get_time();
            const bool dwell_ok = (now_us - last_switch_snapshot) >= SCHED_MIN_STATE_DWELL_US;
            const bool decision_frozen = freeze_windows_snapshot > 0;

            if (quality_ok && hr_consistent && !no_contact)
            {
                dsp_publish_windows++;
                const float bpm_dsp = clampf(peak_bpm, 40.0f, 180.0f);
                if (dsp_publish_windows >= DSP_PUBLISH_WINDOWS)
                {
                    if (!g_has_bpm_ema)
                    {
                        g_bpm_ema = bpm_dsp;
                        g_has_bpm_ema = true;
                    }
                    else
                    {
                        constexpr float kEmaAlpha = 0.35f;
                        g_bpm_ema = (kEmaAlpha * bpm_dsp) + ((1.0f - kEmaAlpha) * g_bpm_ema);
                    }

                    ESP_LOGI(TAG,
                             "DSP_HR=%.2f | state=%d | ac=%.3f ac_hr=%.1f std_hp=%.1f ptp_hp=%.1f",
                             g_bpm_ema,
                             static_cast<int>(state_snapshot),
                             ac_best,
                             ac_best_hr,
                             std_hp,
                             ptp_hp);

                    taskENTER_CRITICAL(&g_telemetry_mux);
                    g_last_hr_report = g_bpm_ema;
                    g_last_hr_source = 0;
                    taskEXIT_CRITICAL(&g_telemetry_mux);
                }
                else
                {
                    ESP_LOGW(TAG,
                             "DSP_HOLD: state=%d peak=%.1f ac_hr=%.1f ac=%.3f",
                             static_cast<int>(state_snapshot),
                             peak_bpm,
                             ac_best_hr,
                             ac_best);
                }
            }
            else
            {
                dsp_publish_windows = 0;
                if (no_contact)
                {
                    ESP_LOGW(TAG,
                             "NO_CONTACT: state=%d ac=%.3f peak_bpm=%.1f std_hp=%.1f ptp_hp=%.1f",
                             static_cast<int>(state_snapshot),
                             ac_best,
                             peak_bpm,
                             std_hp,
                             ptp_hp);
                }
                else
                {
                    ESP_LOGW(TAG,
                             "Low quality window: state=%d reason=%d ac=%.3f peak_bpm=%.1f ac_hr=%.1f std_hp=%.1f ptp_hp=%.1f",
                             static_cast<int>(state_snapshot),
                             static_cast<int>(fail_reason),
                             ac_best,
                             peak_bpm,
                             ac_best_hr,
                             std_hp,
                             ptp_hp);
                }
            }

            if (state_snapshot == SCHED_STATE_NORMAL)
            {
                if (no_contact)
                {
                    bad_windows = 0;
                    gray_windows = 0;
                    good_windows = 0;
                    dsp_publish_windows = 0;
                    continue;
                }

                if (decision_frozen)
                {
                    ESP_LOGW(TAG, "Decision freeze active (windows=%d), hold NORMAL", freeze_windows_snapshot);
                    bad_windows = 0;
                    gray_windows = 0;
                    continue;
                }

                if (severe_motion_fail || (ac_best <= SCHED_AC_HARD) || (difficulty_proxy >= SCHED_DIFF_HARD))
                {
                    bad_windows++;
                    gray_windows = 0;
                }
                else if (mild_fail)
                {
                    gray_windows++;
                    bad_windows = 0;
                }
                else
                {
                    bad_windows = 0;
                    gray_windows = 0;
                }

                if (kAdaptiveEnabled && (bad_windows >= SCHED_BAD_WINDOWS_TO_UP || gray_windows >= GRAY_WINDOWS_TO_UP) && dwell_ok)
                {
                    taskENTER_CRITICAL(&g_state_mux);
                    g_desired_state = SCHED_STATE_HIGH;
                    g_switch_pending = true;
                    taskEXIT_CRITICAL(&g_state_mux);
                    bad_windows = 0;
                    gray_windows = 0;
                    good_windows = 0;
                    cooldown_windows = SCHED_COOLDOWN_WINDOWS;
                }
            }
            else
            {
                if (!no_contact && g_tinyml_ready)
                {
                    float ai_bpm = 0.0f;
                    int8_t y_q = 0;
                    if (run_tinyml_on_features(&ai_bpm, &y_q))
                    {
                        if (!g_has_bpm_ema)
                        {
                            g_bpm_ema = ai_bpm;
                            g_has_bpm_ema = true;
                        }
                        else
                        {
                            constexpr float kAiEmaAlpha = 0.15f;
                            g_bpm_ema = (kAiEmaAlpha * ai_bpm) + ((1.0f - kAiEmaAlpha) * g_bpm_ema);
                        }

                        ESP_LOGI(TAG,
                                 "AI_ASSIST_HR=%.2f | raw_ai=%.2f | y_q=%d | dsp_peak=%.1f ac=%.3f",
                                 g_bpm_ema,
                                 ai_bpm,
                                 static_cast<int>(y_q),
                                 peak_bpm,
                                 ac_best);

                        taskENTER_CRITICAL(&g_telemetry_mux);
                        g_last_hr_report = g_bpm_ema;
                        g_last_hr_source = 1;
                        taskEXIT_CRITICAL(&g_telemetry_mux);
                    }
                }

                if (cooldown_windows > 0)
                {
                    cooldown_windows--;
                }
                else
                {
                    if (decision_frozen)
                    {
                        ESP_LOGW(TAG, "Decision freeze active (windows=%d), hold HIGH", freeze_windows_snapshot);
                        continue;
                    }

                    if (no_contact)
                    {
                        good_windows = SCHED_GOOD_WINDOWS_TO_DOWN;
                    }
                    else if (quality_ok && hr_consistent && ac_best >= SCHED_AC_EASY && difficulty_proxy <= SCHED_DIFF_EASY)
                    {
                        good_windows++;
                    }
                    else
                    {
                        good_windows = 0;
                    }

                    if (kAdaptiveEnabled && good_windows >= SCHED_GOOD_WINDOWS_TO_DOWN && dwell_ok)
                    {
                        taskENTER_CRITICAL(&g_state_mux);
                        g_desired_state = SCHED_STATE_NORMAL;
                        g_switch_pending = true;
                        taskEXIT_CRITICAL(&g_state_mux);
                        good_windows = 0;
                    }
                }
            }
        }
    }
} // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting PPG Scheduler (mode=%d)", static_cast<int>(kRunMode));

    if (kTinyMlEnabledByMode)
    {
        if (!tinyml_init())
        {
            ESP_LOGE(TAG, "TinyML init failed");
            return;
        }
        g_tinyml_ready = true;
    }
    else
    {
        ESP_LOGI(TAG, "TinyML disabled for fixed-normal baseline mode");
    }

    gpio_config_t profiling_gpio_cfg = {};
    profiling_gpio_cfg.pin_bit_mask = (1ULL << PROFILING_FEATURE_GPIO) | (1ULL << PROFILING_INVOKE_GPIO);
    profiling_gpio_cfg.mode = GPIO_MODE_OUTPUT;
    profiling_gpio_cfg.pull_up_en = GPIO_PULLUP_DISABLE;
    profiling_gpio_cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
    profiling_gpio_cfg.intr_type = GPIO_INTR_DISABLE;
    ESP_ERROR_CHECK(gpio_config(&profiling_gpio_cfg));
    ESP_ERROR_CHECK(gpio_set_level(PROFILING_FEATURE_GPIO, 0));
    ESP_ERROR_CHECK(gpio_set_level(PROFILING_INVOKE_GPIO, 0));

    ESP_ERROR_CHECK(i2c_init());
    ESP_ERROR_CHECK(max30102_init());

    printf("timestamp_ms,state,profile,quality,diff,red,ir\n");

    scheduler_state_t initial_state = SCHED_STATE_HIGH;
    if (kRunMode == RUN_MODE_FIXED_NORMAL)
        initial_state = SCHED_STATE_NORMAL;
    ESP_ERROR_CHECK(apply_scheduler_state(initial_state));

    xTaskCreatePinnedToCore(
        inference_task,
        "AI_Task",
        8192,
        nullptr,
        5,
        &inference_task_handle,
        1);

    int64_t last_activity_us = esp_timer_get_time();
    int64_t last_sparse_log_us = 0;
    int consecutive_i2c_errors = 0;

    while (true)
    {
        uint8_t pending = 0;
        if (max30102_fifo_pending(&pending) != ESP_OK)
        {
            consecutive_i2c_errors++;
            add_decision_freeze_windows(DECISION_FREEZE_ON_ERROR_WINDOWS);
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
            if (kAdaptiveEnabled)
            {
                bool do_switch = false;
                scheduler_state_t target_state = SCHED_STATE_NORMAL;

                taskENTER_CRITICAL(&g_state_mux);
                do_switch = g_switch_pending;
                target_state = g_desired_state;
                taskEXIT_CRITICAL(&g_state_mux);

                if (do_switch && apply_scheduler_state(target_state) == ESP_OK)
                {
                    taskENTER_CRITICAL(&g_state_mux);
                    g_switch_pending = false;
                    taskEXIT_CRITICAL(&g_state_mux);
                }
            }

            int64_t now = esp_timer_get_time();

            if (now - last_sparse_log_us >= kSparseLogPeriodUs)
            {
                print_sparse_csv_log(now / 1000LL);
                last_sparse_log_us = now;
            }

            if (now - last_activity_us > 3000000)
            {
                ESP_LOGW(TAG, "No new MAX30102 samples for >3s. Sensor might be sleeping/reset!");
                max30102_recover();
                add_decision_freeze_windows(DECISION_FREEZE_ON_ERROR_WINDOWS);
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
                taskENTER_CRITICAL(&g_telemetry_mux);
                g_last_red = red;
                g_last_ir = ir;
                taskEXIT_CRITICAL(&g_telemetry_mux);

                push_ir_sample(static_cast<float>(ir));
                last_activity_us = esp_timer_get_time();
            }
            else
            {
                consecutive_i2c_errors++;
                add_decision_freeze_windows(DECISION_FREEZE_ON_ERROR_WINDOWS);
                if (consecutive_i2c_errors >= 5)
                {
                    max30102_recover();
                    consecutive_i2c_errors = 0;
                }
                break;
            }
        }

        const int64_t now = esp_timer_get_time();

        if (now - last_sparse_log_us >= kSparseLogPeriodUs)
        {
            print_sparse_csv_log(now / 1000LL);
            last_sparse_log_us = now;
        }

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
