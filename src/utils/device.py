import torch


class DeviceManager:
    @classmethod
    def get_device(cls) -> torch.device:
        """
        Returns the device to be used for computations in the following priority order:
        1. CUDA (GPU) if available
        2. MPS (Apple Silicon GPU) if available
        3. CPU as a fallback

        :return: torch.device object representing the available device.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device.type}")
        return device

    @classmethod
    def check_mps_fallback(cls) -> None:
        """
        Checks if MPS fallback is enabled for PyTorch and prints a warning if not.
        This is useful for ensuring compatibility with Apple Silicon devices.
        """

        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )

        else:
            print("MPS is available and enabled for PyTorch.")
