import os

import torch


class DeviceManager:
    @classmethod
    def get_device(cls) -> torch.device:
        """
        Returns the device to be used for computations in the following priority order:
        1. CUDA (GPU) if available
        2. MPS (Apple Silicon GPU) if available
        3. XPU (Intel GPU) if available
        4. CPU as a fallback

        :return: torch.device object representing the available device.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using dedicated GPU for computations.")
        elif torch.mps.is_available():
            # setup MPS fallback for PyTorch if training on Apple Silicon
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU) for computations.")
        elif torch.xpu.is_available():
            device = torch.device("xpu")
            print("Using XPU (Intel GPU) for computations.")
        else:
            device = torch.device("cpu")
            print("Using CPU for computations. No GPU available.")

        return device

    @classmethod
    def check_gpu_fallback(cls) -> None:
        """
        Checks if GPU fallback is enabled for PyTorch and prints a warning if not.
        This is useful for ensuring compatibility with Apple Silicon devices.
        """

        if not torch.cuda.is_available():
            if not torch.backends.cuda.is_built():
                print(
                    "CUDA not available because the current PyTorch install was not built with CUDA enabled."
                )
            else:
                print(
                    "CUDA not available you do not have a CUDA-enabled device on this machine."
                )

        else:
            print("GPU is available for PyTorch.")

    @classmethod
    def check_mps_fallback(cls) -> None:
        """
        Checks if MPS fallback is enabled for PyTorch and prints a warning if not.
        This is useful for ensuring compatibility with Apple Silicon devices.
        """

        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )

        else:
            print("MPS is available for PyTorch.")

    @classmethod
    def check_xpu_fallback(cls) -> None:
        """
        Checks if XPU fallback is enabled for PyTorch and prints a warning if not.
        This is useful for ensuring compatibility with Intel GPU devices.
        """

        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            print(
                "XPU not available because the current PyTorch install does not support XPU "
                "or the current machine does not have an XPU-enabled device."
            )
        else:
            print("XPU is available for PyTorch.")
