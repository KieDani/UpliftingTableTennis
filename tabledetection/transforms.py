import numpy as np
import scipy as sp
import cv2
import torch

from tabledetection.helper_tabledetection import HEIGHT, WIDTH, KEYPOINT_VISIBLE


class Resize:
    def __init__(self, size):
        '''
        Args:
            size (tuple): Desired output size (width, height).
        '''
        self.size = size

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'coords_list': The coordinates of the table points in the image. List of shape (N, 2) with N table keypoints. Optional
        '''
        image = data['image']
        orig_H, orig_W, C = image.shape
        assert C == 3, "Image must have 3 channels (RGB)"

        # Resize image
        image = cv2.resize(image, self.size)
        if 'coords_list' in data and data['coords_list'] is not None:
            coords_list = data['coords_list']
            # Resize ball coordinates (consider shift to center of pixel with factor 0.5)
            scale_x, scale_y = self.size[0] / orig_W, self.size[1] / orig_H
            coords_list = [((x+0.5) * scale_x - 0.5, (y+0.5) * scale_y - 0.5, v) for (x, y, v) in coords_list]
            data['coords_list'] = coords_list

        data['image'] = image
        return data


class Flip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'coords_list': The coordinates of the table points in the image. List of shape (N, 2) with N table keypoints.
        '''
        # TODO: left and right changes when flipping
        raise RuntimeError("Flip currently not implemented correctly. Do not use")
        if np.random.rand() < self.flip_prob:
            image = data['image']
            coords_list = data['coords_list']

            # Flip image
            image = cv2.flip(image, 1)
            # Flip ball coordinates
            coords_list = [(image.shape[1] - x - 1, y) for (x, y) in coords_list]

            data['image'] = image
            data['coords_list'] = coords_list
        return data


class Translation:
    def __init__(self, prob=0.5, max_transl=0.2):
        self.prob = prob
        self.max_transl = max_transl

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y).
        '''
        if np.random.rand() < self.prob:
            image = data['image']
            table_coords = data['coords_list']

            h, w = image.shape[:2]
            transl_x = np.random.randint(-int(self.max_transl * w), int(self.max_transl * w))
            transl_y = np.random.randint(-int(self.max_transl * h), int(self.max_transl * h))

            # transformation matrix
            M = np.float32([[1, 0, transl_x], [0, 1, transl_y]])

            # Perform the affine transformation
            image = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
            # Adjust coordinates
            table_coords[:, 0] += transl_x
            table_coords[:, 1] += transl_y
            table_coords[:, 2] = np.logical_and(table_coords[:, 2] == KEYPOINT_VISIBLE,
                                 np.logical_and((np.logical_and(table_coords[:, 0] < w, 0 <= table_coords[:, 0])),
                                 np.logical_and(0 <= table_coords[:, 1], table_coords[:, 1] < h)))

            data['image'] = image
            data['coords_list'] = table_coords
        return data


class Rotation:
    def __init__(self, prob=0.5, max_rot=10):
        self.prob = prob
        self.max_rot = max_rot

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y).
        '''
        if np.random.rand() < self.prob:
            image = data['image']
            table_coords = data['coords_list']

            h, w = image.shape[:2]
            rot = np.random.randint(-self.max_rot, self.max_rot)

            # transformation matrix
            rot_center = (w / 2, h / 2)

            # Get the 2x3 rotation matrix
            # OpenCV's getRotationMatrix2D uses angle in degrees. Positive is counter-clockwise.
            M = cv2.getRotationMatrix2D(rot_center, rot, 1.0)

            # Perform the affine transformation
            image = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
            # Adjust coordinates
            keypoints = table_coords[:, :2]
            keypoints_hom = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
            # Apply the transformation matrix
            adjusted_keypoints_transposed = M @ keypoints_hom.T
            adjusted_keypoints = adjusted_keypoints_transposed.T
            table_coords[:, :2] = adjusted_keypoints
            table_coords[:, 2] = np.logical_and(table_coords[:, 2] == KEYPOINT_VISIBLE,
                                 np.logical_and((np.logical_and(table_coords[:, 0] < w, 0 <= table_coords[:, 0])),
                                 np.logical_and(0 <= table_coords[:, 1], table_coords[:, 1] < h)))

            data['image'] = image
            data['coords_list'] = table_coords
        return data


class Crop:
    def __init__(self, prob=0.5, min_fraction=0.8):
        self.prob = prob
        self.min_frac = min_fraction

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y).
        '''
        if np.random.rand() < self.prob:
            if not (0.0 <= self.min_frac <= 1.0):
                raise ValueError("min_fraction must be between 0.0 and 1.0.")
            if self.min_frac == 0.0:
                return data

            image = data['image']
            table_coords = data['coords_list']
            h, w = image.shape[:2]

            # 1. Randomly choose a scale factor for the crop
            # The crop will be between min_fraction * original_size and 1.0 * original_size
            scale_factor = np.random.uniform(self.min_frac, 1.0)

            # 2. Calculate the crop dimensions while preserving aspect ratio
            crop_h = int(np.round(h * scale_factor))
            crop_w = int(np.round(w * scale_factor))

            # 3. Randomly choose the top-left corner (x_min, y_min)
            # x_min can be between 0 and (width - crop_width)
            x_min = np.random.randint(0, w - crop_w + 1)
            # y_min can be between 0 and (height - crop_height)
            y_min = np.random.randint(0, h - crop_h + 1)

            x_max = x_min + crop_w
            y_max = y_min + crop_h

            # 4. Perform the crop using NumPy slicing
            cropped_image = image[y_min:y_max, x_min:x_max]
            h, w = cropped_image.shape[:2]
            # Adjust coordinates and visibility
            table_coords[:, 0] -= x_min
            table_coords[:, 1] -= y_min
            table_coords[:, 2] = np.logical_and(table_coords[:, 2] == KEYPOINT_VISIBLE,
                                 np.logical_and((np.logical_and(table_coords[:, 0] < w, 0 <= table_coords[:, 0])),
                                 np.logical_and(0 <= table_coords[:, 1], table_coords[:, 1] < h)))

            data['image'] = cropped_image
            data['coords_list'] = table_coords

        return data


class ColorJitter:
    def __init__(self, prob=0.5, brightness_factor=0.2, contrast_factor=0.2, saturation_factor=0.2, hue_factor=0.1):
        self.prob = prob
        self.max_brightness = brightness_factor
        self.max_contrast = contrast_factor
        self.max_saturation = saturation_factor
        self.max_hue = hue_factor

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y).
        '''
        if np.random.rand() < self.prob:
            image = data['image']

            brightness = np.random.uniform(0.0, self.max_brightness)
            contrast = np.random.uniform(0.0, self.max_contrast)
            saturation = np.random.uniform(0.0, self.max_saturation)
            hue = np.random.uniform(0.0, self.max_hue)
            image = apply_color_jitter(image, brightness, contrast, saturation, hue)
            data['image'] = image
        return data


def apply_color_jitter(image, brightness_factor=0.2, contrast_factor=0.2, saturation_factor=0.2, hue_factor=0.1):
    """
    Applies color jitter (brightness, contrast, saturation, hue) to an image using OpenCV.

    Args:
        image (np.ndarray): Input image in BGR format (H, W, C), typically uint8.
        brightness_factor (float): Max factor for random brightness adjustment (e.g., 0.2 means +/- 20%).
        contrast_factor (float): Max factor for random contrast adjustment (e.g., 0.2 means +/- 20%).
        saturation_factor (float): Max factor for random saturation adjustment (e.g., 0.2 means +/- 20%).
        hue_factor (float): Max factor for random hue adjustment (e.g., 0.1 means +/- 10% of 179 for OpenCV hue range).

    Returns:
        np.ndarray: Image with color jitter applied, in BGR format (uint8).
    """
    # 1. Convert to float32 for calculations (recommended)
    image_float = image.astype(np.float32) / 255.0 # Normalize to [0, 1]

    # --- Brightness Adjustment (additive) ---
    brightness_delta = (np.random.rand() * 2 - 1) * brightness_factor # Random between -factor and +factor
    image_float = image_float + brightness_delta
    image_float = np.clip(image_float, 0, 1) # Clip values to [0, 1]

    # --- Contrast Adjustment (multiplicative) ---
    contrast_scale = 1.0 + (np.random.rand() * 2 - 1) * contrast_factor # Random between (1-factor) and (1+factor)
    image_float = image_float * contrast_scale
    image_float = np.clip(image_float, 0, 1)

    # --- Saturation and Hue Adjustment (requires HSV conversion) ---
    image_hsv = cv2.cvtColor((image_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    image_hsv_float = image_hsv.astype(np.float32)

    # Saturation
    saturation_scale = 1.0 + (np.random.rand() * 2 - 1) * saturation_factor
    image_hsv_float[:, :, 1] = image_hsv_float[:, :, 1] * saturation_scale
    image_hsv_float[:, :, 1] = np.clip(image_hsv_float[:, :, 1], 0, 255) # Clip S to [0, 255]

    # Hue
    # OpenCV's H channel range is 0-179.
    # Adjusting hue by adding a random delta
    hue_delta = (np.random.rand() * 2 - 1) * (hue_factor * 179.0) # Delta based on 0-179 range
    image_hsv_float[:, :, 0] = image_hsv_float[:, :, 0] + hue_delta
    # Hue wraps around (0 and 179 are red). Use modulo for wrapping.
    image_hsv_float[:, :, 0] = np.fmod(np.fmod(image_hsv_float[:, :, 0], 180.0) + 180.0, 180.0)
    # No clipping needed for hue after modulo, as it naturally wraps.

    # Convert back to BGR and uint8
    image_hsv_uint8 = image_hsv_float.astype(np.uint8)
    jittered_image_bgr = cv2.cvtColor(image_hsv_uint8, cv2.COLOR_HSV2BGR)

    return jittered_image_bgr


class PerspectiveTransform:
    def __init__(self, prob=0.5, max_warp_factor=0.1):
        """
        Initializes the PerspectiveTransform augmentation.

        Args:
            prob (float): The probability of applying the transformation.
            max_warp_factor (float): A factor controlling the magnitude of the perspective distortion.
                                     It determines the maximum random displacement of the image corners
                                     as a fraction of the image's dimensions.
        """
        self.prob = prob
        self.max_warp_factor = max_warp_factor

    def __call__(self, data):
        """
        Applies a random perspective transformation to the image and keypoints.

        Args:
            data (dict): A dictionary containing the data to be augmented.
                - 'image' (np.ndarray): The input image with shape (H, W, C).
                - 'coords_list' (np.ndarray): A NumPy array of keypoint coordinates with shape (N, 3),
                                              where each row is (x, y, visibility).

        Returns:
            dict: The augmented data dictionary.
        """
        if np.random.rand() < self.prob:
            image = data['image']
            table_coords = data['coords_list']
            h, w = image.shape[:2]

            # 1. Define the four corners of the original image (source points).
            src_points = np.float32([
                [0, 0],  # Top-left
                [w - 1, 0],  # Top-right
                [0, h - 1],  # Bottom-left
                [w - 1, h - 1]  # Bottom-right
            ])

            # 2. Define the destination points by randomly displacing the source corners.
            max_dx = self.max_warp_factor * w
            max_dy = self.max_warp_factor * h

            # Generate random horizontal and vertical shifts for each corner.
            dx = np.random.uniform(-max_dx, max_dx, 4)
            dy = np.random.uniform(-max_dy, max_dy, 4)

            dst_points = src_points + np.vstack([dx, dy]).T
            dst_points = dst_points.astype(np.float32)

            # 3. Compute the 3x3 perspective transformation matrix.
            M = cv2.getPerspectiveTransform(src_points, dst_points)

            # 4. Apply the perspective warp to the image.
            warped_image = cv2.warpPerspective(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)  # Fill new areas with black
            )

            # 5. Transform the keypoint coordinates.
            if table_coords is not None and len(table_coords) > 0:
                keypoints = table_coords[:, :2]

                # Convert to homogeneous coordinates [x, y, 1] for perspective transformation.
                keypoints_hom = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))

                # Apply the transformation matrix: M * [x, y, 1]^T = [x', y', w']^T
                transformed_hom = M @ keypoints_hom.T

                # Convert back to Cartesian coordinates by dividing by the third component (w').
                # Add a small epsilon to avoid division by zero.
                w_prime = transformed_hom[2, :] + 1e-8
                new_x = transformed_hom[0, :] / w_prime
                new_y = transformed_hom[1, :] / w_prime

                adjusted_keypoints = np.vstack((new_x, new_y)).T
                table_coords[:, :2] = adjusted_keypoints

                # 6. Update the visibility status of each keypoint.
                # A keypoint is now only visible if it was visible before AND is still inside the image bounds.
                is_inside = (adjusted_keypoints[:, 0] >= 0) & (adjusted_keypoints[:, 0] < w) & \
                            (adjusted_keypoints[:, 1] >= 0) & (adjusted_keypoints[:, 1] < h)

                # Assume KEYPOINT_VISIBLE is a constant like 1 or True.
                table_coords[:, 2] = np.logical_and(table_coords[:, 2] == KEYPOINT_VISIBLE, is_inside)

            # Update the dictionary with the transformed data.
            data['image'] = warped_image
            data['coords_list'] = table_coords

        return data


class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'coords_list': The coordinates of the table points in the image. List of shape (N, 2) with N table keypoints. Optional
        '''
        image = data['image']
        image = image / 255.0
        image = (image - self.mean) / self.std
        data['image'] = image
        return data


class UnnormalizeImage:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
        '''
        image = data['image']
        image = np.clip((image * np.array(self.std)[:, None, None] + np.array(self.mean)[:, None, None]) * 255.0, 0, 255).astype(np.uint8)
        data['image'] = image
        return data


class ResizeTorch:
    def __init__(self, size):
        '''
        Args:
            size (tuple): Desired output size (width, height).
        '''
        self.size = size

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (B, C, H, W).
                - 'coords_list': The coordinates of the table points in the image. Shape (B, N, 2). Optional

        '''
        image = data['image']
        H, W = image.shape[2], image.shape[3]
        image = torch.nn.functional.interpolate(image, size=self.size, mode='bilinear', align_corners=False)
        if 'coords_list' in data and data['coords_list'] is not None:
            coords_list = data['coords_list']
            coords_list[..., 0] = (coords_list[:, :, 0] + 0.5) * self.size[0] / W - 0.5
            coords_list[..., 1] = (coords_list[:, :, 1] + 0.5) * self.size[1] / H - 0.5
            data['coords_list'] = coords_list
        data['image'] = image
        return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'coords_list': The coordinates of the table points in the image. List of shape (N, 2) with N table keypoints.
        '''
        for transform in self.transforms:
            data = transform(data)
        return data


def get_transform(mode, resolution):
    '''
    Get the transform function based on the mode.
    Args:
        mode (str): Mode of the transform ('train', 'val', 'test').
        resolution (tuple): Desired image size (width, height).
    '''
    assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val' or 'test'"
    assert len(resolution) == 2, "Resolution must be a tuple of (width, height)"
    if mode == 'train':
        return Compose([
            # Flip(flip_prob=0.5),
            PerspectiveTransform(prob=0.5, max_warp_factor=0.15),
            Rotation(),
            Translation(),
            Crop(),
            Resize(resolution),
            ColorJitter(),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif mode == 'val':
        return Compose([
            Resize(resolution),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif mode == 'test':
        return Compose([
            Resize(resolution),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Mode must be 'train', 'val' or 'test'")


plot_transforms = Compose([
    UnnormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resize_transform = ResizeTorch((WIDTH, HEIGHT))

