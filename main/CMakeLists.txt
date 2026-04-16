idf_component_register(
    SRCS 
        "ppg_hr_tinyml.cpp"
        "ppg_hr_mlp_int8.c"
    INCLUDE_DIRS "."
    REQUIRES esp-dsp esp_driver_gpio esp_driver_i2c driver esp_timer
)
