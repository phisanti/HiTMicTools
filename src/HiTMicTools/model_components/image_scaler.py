# Type annotation imports
from typing import Optional

# Standard imports
import torch

# Local imports
from HiTMicTools.utils import get_device


class ImageScaler:
    """Image scaling utility for microscopy data processing.

    Provides multiple normalization methods for 4D image tensors (batch, channel, height, width):
    - range01: Percentile-based normalization to [0,1] range
    - zscore: Z-score standardization
    - combined: Percentile normalization followed by z-score
    - fixed_range: Bit-depth based normalization
    - equalize: Histogram equalization

    Attributes:
        device (torch.device): Device for computations
        dtype (torch.dtype): Data type for tensors
        scale_method (str): Selected normalization method
        pmin (float): Lower percentile for range01/combined methods
        pmax (float): Upper percentile for range01/combined methods
        clip (bool): Whether to clip values to [0,1]
        fixed_min (int): Minimum value for fixed range scaling
        fixed_max (int): Maximum value for fixed range scaling
    """

    def __init__(
        self,
        scale_method: str = "range01",
        bit_depth: Optional[int] = None,
        pmin: Optional[float] = None,
        pmax: Optional[float] = None,
        clip: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize ImageScaler with specified normalization parameters.

        Args:
            scale_method: Normalization method to use
            bit_depth: Image bit depth for fixed_range method
            pmin: Lower percentile for range01/combined methods (default: 1)
            pmax: Upper percentile for range01/combined methods (default: 99.8)
            clip: Whether to clip normalized values to [0,1]
            device: Computation device (default: GPU if available)
            dtype: Tensor data type (default: float32)

        Raises:
            ValueError: If parameters don't match selected scale_method
        """
        # Device handling
        self.device = device or get_device()
        self.dtype = dtype
        self.scale_method = scale_method

        # Method-specific parameters
        if scale_method == "range01" or scale_method == "combined":
            if pmin is None or pmax is None:
                self.pmin = 1
                self.pmax = 99.8
            else:
                if not 0 <= pmin < pmax <= 100:
                    raise ValueError("pmin/pmax must be 0 <= pmin < pmax <= 100")
                self.pmin = pmin
                self.pmax = pmax
            self.clip = clip

        elif scale_method == "fixed_range":
            if bit_depth is None:
                raise ValueError("bit_depth required for fixed_range scaling")
            if not 1 <= bit_depth <= 32:
                raise ValueError("bit_depth must be between 1 and 32")
            self.fixed_min = 0
            self.fixed_max = 2**bit_depth - 1

        # Initialize tracking attributes
        self.mi = None
        self.ma = None
        self.mean = None
        self.std = None

    def scale_fixed_range(self, image: torch.Tensor) -> torch.Tensor:
        """
        Fixed range normalization using camera's bit depth range.

        Args:
            image (torch.Tensor): Input tensor of shape (batch, channel, height, width)

        Returns:
            torch.Tensor: Normalized tensor between 0 and 1
        """
        return (image - self.fixed_min) / (self.fixed_max - self.fixed_min)

    def normalize_percentile_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Percentile-based image normalization for 4D standard image format (batch, channel, height, width).

        The function also stores the percentile values used for normalization to scale back predictions.

        Returns:
            torch.Tensor: Normalized tensor of shape (batch_size, channel, height, width).
        """
        images = images.to(self.dtype)
        # Calculate quantiles along height dimension first
        temp = torch.quantile(images, self.pmin / 100, dim=2, keepdim=True)
        # Then along width dimension
        self.mi = torch.quantile(temp, self.pmin / 100, dim=3, keepdim=True)

        temp = torch.quantile(images, self.pmax / 100, dim=2, keepdim=True)
        self.ma = torch.quantile(temp, self.pmax / 100, dim=3, keepdim=True)

        eps = torch.finfo(self.dtype).eps
        images = (images - self.mi) / (self.ma - self.mi + eps)
        if self.clip:
            images = torch.clamp(images, 0, 1)
        return images

    def scale_zscore(self, image):
        """
        Z-score normalization of the image for 4D standard image format (batch, channel, height, width).
        The function also stores the mean and standard deviation of each image to scale back predictions.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The normalized image tensor.
        """
        self.mean = torch.mean(image, dim=(2, 3), keepdim=True)
        self.std = torch.std(image, dim=(2, 3), keepdim=True)
        return (image - self.mean) / (self.std + 1e-6)

    def scale_combined(self, image):
        """
        Combined normalization of the image for 4D standard image format (batch, channel, height, width).
        The function also stores the percentile values used for normalization to scale back predictions.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The normalized image.
        """
        image = self.normalize_percentile_batch(image)
        image = self.scale_zscore(image)

        return image

    def scale_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Scale image using specified method for 4D images (batch, channel, height, width).

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The scaled image tensor.
        """
        # Check input shape images, or raise error
        if image.ndim != 4:
            raise ValueError("Input image must be 4D (batch, channel, height, width)")

        if self.scale_method == "range01":
            return self.normalize_percentile_batch(image)
        elif self.scale_method == "zscore":
            return self.scale_zscore(image)
        elif self.scale_method == "combined":
            return self.scale_combined(image)
        elif self.scale_method == "fixed_range":
            return self.scale_fixed_range(image)

    def rescale_image(self, image: torch.Tensor, scale_method: str) -> torch.Tensor:
        """
        Rescale the image tensor to the original range. The image has to be scaled with this class in order to be rescaled.

        Args:
            image (torch.Tensor): The input image tensor.
            scale_method (str): The scaling method used during normalization.

        Returns:
            torch.Tensor: The rescaled image tensor.
        """
        if image.ndim != 4:
            raise ValueError("Input image must be 4D (batch, channel, height, width)")

        if scale_method == "range01":
            image = image * (self.ma - self.mi) + self.mi
        elif scale_method == "zscore":
            image = image * self.std + self.mean
        elif scale_method == "combined":
            image = image * self.std + self.mean
            image = image * (self.ma - self.mi) + self.mi
        elif scale_method == "fixed_range":
            image = image * (self.fixed_max - self.fixed_min) + self.fixed_min

        return image


if __name__ == "__main__":
    # Create test data
    test_image = torch.rand(2, 3, 64, 64) * 255

    def compute_error(original, reconstructed):
        mse = torch.mean((original - reconstructed) ** 2)
        mae = torch.mean(torch.abs(original - reconstructed))
        return {"MSE": mse.item(), "MAE": mae.item()}

    # Test range01 scaling
    scaler_range = ImageScaler(scale_method="range01", clip=False)
    scaled_range = scaler_range.scale_image(test_image)
    rescaled_range = scaler_range.rescale_image(scaled_range, "range01")
    print(f"Scaled range for range01: {scaled_range.min()}, {scaled_range.max()}")
    print(f"Rescaled range for range01: {rescaled_range.min()}, {rescaled_range.max()}")
    print(
        f"Range01 test passed: {torch.allclose(test_image, rescaled_range, rtol=1e-2)}"
    )

    # Test zscore scaling
    scaler_zscore = ImageScaler(scale_method="zscore")
    scaled_zscore = scaler_zscore.scale_image(test_image)
    rescaled_zscore = scaler_zscore.rescale_image(scaled_zscore, "zscore")
    print(
        f"Z-score test passed: {torch.allclose(test_image, rescaled_zscore, rtol=1e-2)}"
    )

    # Test combined scaling
    scaler_combined = ImageScaler(scale_method="combined", clip=False)
    scaled_combined = scaler_combined.scale_image(test_image)
    rescaled_combined = scaler_combined.rescale_image(scaled_combined, "combined")
    print(
        f"Scaled range for combined method: {scaled_range.min()}, {scaled_range.max()}"
    )
    print(
        f"Rescaled range for combined method: {rescaled_range.min()}, {rescaled_range.max()}"
    )

    print(
        f"Combined test passed: {torch.allclose(test_image, rescaled_combined, rtol=1e-2)}"
    )

    # Test fixed range scaling
    scaler_fixed = ImageScaler(scale_method="fixed_range", bit_depth=8)
    scaled_fixed = scaler_fixed.scale_image(test_image)
    rescaled_fixed = scaler_fixed.rescale_image(scaled_fixed, "fixed_range")
    print(
        f"Fixed range test passed: {torch.allclose(test_image, rescaled_fixed, rtol=1e-4)}"
    )

    # Test each method
    methods = ["range01", "zscore", "combined", "fixed_range"]
    for method in methods:
        kwargs = {"bit_depth": 8} if method == "fixed_range" else {}
        scaler = ImageScaler(scale_method=method, **kwargs)
        scaled = scaler.scale_image(test_image)
        rescaled = scaler.rescale_image(scaled, method)
        errors = compute_error(test_image, rescaled)
        print(f"\n{method.upper()} Method:")
        print(f"Scaled range: [{scaled.min():.4f}, {scaled.max():.4f}]")
        print(f"Reconstruction errors: {errors}")
