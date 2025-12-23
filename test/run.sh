# ncu --set full --target-processes all python 57_conv_transposed_2D__square_input__square_kernel.py


# ncu --nvtx --nvtx-include "TARGET_FORWARD" --set full --target-processes all python 57_conv_transposed_2D__square_input__square_kernel.py

ncu --csv --profile-from-start off --set full --target-processes all python 57_conv_transposed_2D__square_input__square_kernel.py



