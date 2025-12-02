import numpy as np
import scipy as sp
import cv2
import torch

from balldetection.helper_balldetection import HEIGHT, WIDTH, BALL_VISIBLE, BALL_INVISIBLE


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
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y). Optional.
        '''
        image = data['image']
        orig_H, orig_W, C = image.shape
        assert C == 3, "Image must have 3 channels (RGB), Channel first"
        if 'ball_coords' in data and data['ball_coords'] is not None:
            ball_coords = data['ball_coords']

        # Resize image
        image = cv2.resize(image, self.size)
        if 'prev_image' in data and data['prev_image'] is not None:
            prev_image = data['prev_image']
            prev_image = cv2.resize(prev_image, self.size)
            data['prev_image'] = prev_image
        if 'next_image' in data and data['next_image'] is not None:
            next_image = data['next_image']
            next_image = cv2.resize(next_image, self.size)
            data['next_image'] = next_image
        if 'ball_coords' in data and data['ball_coords'] is not None:
            # Resize ball coordinates (consider shift to center of pixel with factor 0.5)
            ball_x, ball_y = ball_coords
            ball_x, ball_y = ball_x + 0.5, ball_y + 0.5
            ball_x = ball_x * self.size[0] / orig_W - 0.5
            ball_y = ball_y * self.size[1] / orig_H - 0.5
            ball_coords = (ball_x, ball_y)
            data['ball_coords'] = ball_coords

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
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y).
        '''
        if np.random.rand() < self.flip_prob:
            image = data['image']
            ball_coords = data['ball_coords']

            # Flip image
            image = cv2.flip(image, 1)
            # Flip ball coordinates
            ball_x, ball_y = ball_coords
            ball_x = image.shape[1] - ball_x - 1
            ball_coords = (ball_x, ball_y)
            if 'prev_image' in data and data['prev_image'] is not None:
                prev_image = data['prev_image']
                prev_image = cv2.flip(prev_image, 1)
                data['prev_image'] = prev_image
            if 'next_image' in data and data['next_image'] is not None:
                next_image = data['next_image']
                next_image = cv2.flip(next_image, 1)
                data['next_image'] = next_image

            data['image'] = image
            data['ball_coords'] = ball_coords
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
            ball_coords = data['ball_coords']

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
            ball_x, ball_y = ball_coords
            ball_x += transl_x
            ball_y += transl_y
            vis = BALL_VISIBLE if data['visibility'] == BALL_VISIBLE and (0 <= ball_x < w) and (0 <= ball_y < h) else BALL_INVISIBLE
            ball_coords = (ball_x, ball_y)
            if 'prev_image' in data and data['prev_image'] is not None:
                prev_image = data['prev_image']
                prev_image = cv2.warpAffine(prev_image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
                data['prev_image'] = prev_image
            if 'next_image' in data and data['next_image'] is not None:
                next_image = data['next_image']
                next_image =  cv2.warpAffine(next_image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
                data['next_image'] = next_image

            data['image'] = image
            data['ball_coords'] = ball_coords
            data['visibility'] = vis
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
            ball_coords = data['ball_coords']

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
            ball_x, ball_y = ball_coords
            keypoints = np.asarray([ball_x, ball_y])[None]
            keypoints_hom = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
            # Apply the transformation matrix
            adjusted_keypoints_transposed = M @ keypoints_hom.T
            adjusted_keypoints = adjusted_keypoints_transposed.T[0]
            ball_x = adjusted_keypoints[0]
            ball_y = adjusted_keypoints[1]

            vis = BALL_VISIBLE if data['visibility'] == BALL_VISIBLE and (0 <= ball_x < w) and (0 <= ball_y < h) else BALL_INVISIBLE
            ball_coords = (ball_x, ball_y)
            if 'prev_image' in data and data['prev_image'] is not None:
                prev_image = data['prev_image']
                prev_image = cv2.warpAffine(prev_image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
                data['prev_image'] = prev_image
            if 'next_image' in data and data['next_image'] is not None:
                next_image = data['next_image']
                next_image =  cv2.warpAffine(next_image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
                data['next_image'] = next_image

            data['image'] = image
            data['ball_coords'] = ball_coords
            data['visibility'] = vis
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
            ball_coords = data['ball_coords']
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
            ball_x, ball_y = ball_coords
            ball_x -= x_min
            ball_y -= y_min
            vis = BALL_VISIBLE if data['visibility'] == BALL_VISIBLE and (0 <= ball_x < w) and (0 <= ball_y < h) else BALL_INVISIBLE
            ball_coords = (ball_x, ball_y)

            if 'prev_image' in data and data['prev_image'] is not None:
                prev_image = data['prev_image']
                prev_image = prev_image[y_min:y_max, x_min:x_max]
                data['prev_image'] = prev_image
            if 'next_image' in data and data['next_image'] is not None:
                next_image = data['next_image']
                next_image =  next_image[y_min:y_max, x_min:x_max]
                data['next_image'] = next_image

            data['image'] = cropped_image
            data['ball_coords'] = ball_coords
            data['visibility'] = vis

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

            if 'prev_image' in data and data['prev_image'] is not None:
                prev_image = data['prev_image']
                prev_image = apply_color_jitter(prev_image, brightness, contrast, saturation, hue)
                data['prev_image'] = prev_image
            if 'next_image' in data and data['next_image'] is not None:
                next_image = data['next_image']
                next_image =  apply_color_jitter(next_image, brightness, contrast, saturation, hue)
                data['next_image'] = next_image

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



class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        '''
        Args:
            data (dict): Dictionary containing the image and ball coordinates.
                - 'image': The input image. Shape (H, W, C).
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y).
        '''
        image = data['image']
        image = image / 255.0
        image = (image - self.mean) / self.std
        data['image'] = image
        if 'prev_image' in data and data['prev_image'] is not None:
            prev_image = data['prev_image']
            prev_image = prev_image / 255.0
            prev_image = (prev_image - self.mean) / self.std
            data['prev_image'] = prev_image
        if 'next_image' in data and data['next_image'] is not None:
            next_image = data['next_image']
            next_image = next_image / 255.0
            next_image = (next_image - self.mean) / self.std
            data['next_image'] = next_image
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
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
        '''
        image = data['image']
        image = np.clip((image * np.array(self.std)[:, None, None] + np.array(self.mean)[:, None, None]) * 255.0, 0, 255).astype(np.uint8)
        data['image'] = image
        if 'prev_image' in data and data['prev_image'] is not None:
            prev_image = data['prev_image']
            prev_image = np.clip((prev_image * np.array(self.std)[:, None, None] + np.array(self.mean)[:, None, None]) * 255.0, 0, 255).astype(np.uint8)
            data['prev_image'] = prev_image
        if 'next_image' in data and data['next_image'] is not None:
            next_image = data['next_image']
            next_image = np.clip((next_image * np.array(self.std)[:, None, None] + np.array(self.mean)[:, None, None]) * 255.0, 0, 255).astype(np.uint8)
            data['next_image'] = next_image
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
                - 'ball_coords': The coordinates of the ball in the image. Shape (B, 2). Optional

        '''
        image = data['image']
        H, W = image.shape[2], image.shape[3]
        image = torch.nn.functional.interpolate(image, size=self.size, mode='bilinear', align_corners=False)
        if 'ball_coords' in data:
            ball_coords = data['ball_coords']
            ball_x, ball_y = ball_coords[:, 0], ball_coords[:, 1]
            ball_x = (ball_x + 0.5) * self.size[0] / W - 0.5
            ball_y = (ball_y + 0.5) * self.size[1] / H - 0.5
            ball_coords = torch.stack((ball_x, ball_y), dim=1)
            data['ball_coords'] = ball_coords
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
                - 'prev_image': The previous image. Shape (H, W, C). Optional
                - 'next_image': The next image. Shape (H, W, C). Optional
                - 'ball_coords': The coordinates of the ball in the image. Tupel (x, y).
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
            Flip(flip_prob=0.5),
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


