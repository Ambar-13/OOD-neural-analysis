"""
Image perturbations for the out-of-distribution test.

Every function takes float32 images of shape (n, height, width, 3) in [0, 1]
and returns the same shape and range. build_conditions makes the full set.

The Allen natural scenes are grayscale, so hue, saturation and grayscale are
identities on them and build_conditions leaves them out.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from scipy.ndimage import convolve, gaussian_filter, rotate
from scipy.ndimage import zoom as scipy_zoom


def _as_pil(img):
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def _from_pil(pil):
    return np.asarray(pil, dtype=np.float32) / 255.0


def _each(images, fn):
    return np.stack([np.clip(fn(img), 0, 1).astype(np.float32) for img in images])


def contrast(images, factor):
    return _each(images, lambda im: _from_pil(ImageEnhance.Contrast(_as_pil(im)).enhance(factor)))


def brightness(images, factor):
    return _each(images, lambda im: _from_pil(ImageEnhance.Brightness(_as_pil(im)).enhance(factor)))


def saturation(images, factor):
    return _each(images, lambda im: _from_pil(ImageEnhance.Color(_as_pil(im)).enhance(factor)))


def hue_shift(images, shift):
    def fn(im):
        hsv = rgb_to_hsv(im)
        hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
        return hsv_to_rgb(hsv)
    return _each(images, fn)


def grayscale(images):
    return _each(images, lambda im: _from_pil(_as_pil(im).convert("L").convert("RGB")))


def gaussian_blur(images, sigma):
    return _each(images, lambda im: _from_pil(_as_pil(im).filter(ImageFilter.GaussianBlur(radius=sigma))))


def _convolve_rgb(im, kernel):
    return np.stack([convolve(im[..., c], kernel, mode="reflect") for c in range(3)], axis=-1)


def defocus_blur(images, radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(float)
    kernel /= kernel.sum()
    return _each(images, lambda im: _convolve_rgb(im, kernel))


def motion_blur(images, size, angle):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0
    kernel = rotate(kernel, angle, reshape=False, order=1)
    kernel /= kernel.sum()
    return _each(images, lambda im: _convolve_rgb(im, kernel))


def zoom_blur(images, zoom_factor, n_steps=10):
    """Average of the image zoomed in by steps from 1 to zoom_factor."""
    h, w = images.shape[1:3]

    def fn(im):
        out = np.zeros_like(im)
        for step in range(n_steps):
            alpha = 1.0 + (zoom_factor - 1.0) * step / n_steps
            zoomed = scipy_zoom(im, (alpha, alpha, 1), order=1)
            top, left = (zoomed.shape[0] - h) // 2, (zoomed.shape[1] - w) // 2
            out += zoomed[top:top + h, left:left + w] / n_steps
        return out
    return _each(images, fn)


def gaussian_noise(images, std, rng):
    return np.clip(images + rng.randn(*images.shape).astype(np.float32) * std, 0, 1).astype(np.float32)


def shot_noise(images, scale, rng):
    return _each(images, lambda im: rng.poisson(im * 255 * scale) / (255 * scale))


def impulse_noise(images, amount, rng):
    def fn(im):
        out = im.copy()
        out[rng.rand(*im.shape[:2]) < amount / 2] = 1.0
        out[rng.rand(*im.shape[:2]) < amount / 2] = 0.0
        return out
    return _each(images, fn)


def speckle_noise(images, std, rng):
    noise = rng.randn(*images.shape).astype(np.float32) * std
    return np.clip(images * (1 + noise), 0, 1).astype(np.float32)


def snow(images, severity, rng):
    def fn(im):
        flakes = gaussian_filter((rng.rand(*im.shape[:2]) < 0.1 * severity).astype(float), sigma=1.0)[..., None]
        return im * (1 - flakes * 0.7) + flakes * 0.9
    return _each(images, fn)


def frost(images, severity, rng):
    def fn(im):
        layer = gaussian_filter(rng.rand(*im.shape[:2]) * severity, sigma=2.0)[..., None]
        return im * (1 - layer * 0.5) + layer * 0.8
    return _each(images, fn)


def fog(images, severity):
    return np.clip(images * (1 - severity) + 0.8 * severity, 0, 1).astype(np.float32)


def is_grayscale(images, tol=1e-3):
    return float(np.abs(images[..., 0] - images[..., 1]).max()) < tol and \
        float(np.abs(images[..., 0] - images[..., 2]).max()) < tol


def build_conditions(images, rng):
    """All conditions as {name: images}. Conditions that leave the images unchanged are dropped."""
    conds = {}
    for f in [0.1, 0.3, 0.5, 1.5, 2.5, 4.0]:
        conds[f"contrast_{f}"] = contrast(images, f)
        conds[f"brightness_{f}"] = brightness(images, f)
    if not is_grayscale(images):
        for f in [0.0, 0.2, 0.5, 1.5, 2.5, 4.0]:
            conds[f"saturation_{f}"] = saturation(images, f)
        for h in [-0.5, -0.3, -0.15, 0.15, 0.3, 0.5]:
            conds[f"hue_{h:+.2f}"] = hue_shift(images, h)
        conds["grayscale"] = grayscale(images)
    for s in [1, 2, 4, 6, 10]:
        conds[f"blur_gaussian_s{s}"] = gaussian_blur(images, float(s))
    for r in [2, 4, 6]:
        conds[f"blur_defocus_r{r}"] = defocus_blur(images, r)
    for size in [5, 10, 15]:
        for angle in [0, 45]:
            conds[f"blur_motion_s{size}_a{angle}"] = motion_blur(images, size, angle)
    for z in [1.1, 1.2, 1.3]:
        conds[f"blur_zoom_{z}"] = zoom_blur(images, z)
    for sd in [0.02, 0.05, 0.10, 0.15, 0.25]:
        conds[f"noise_gaussian_{sd:.2f}"] = gaussian_noise(images, sd, rng)
    for sc in [0.5, 1.0, 2.0]:
        conds[f"noise_shot_{sc}"] = shot_noise(images, sc, rng)
    for a in [0.01, 0.03, 0.05]:
        conds[f"noise_impulse_{a:.2f}"] = impulse_noise(images, a, rng)
    for sd in [0.05, 0.10, 0.20]:
        conds[f"noise_speckle_{sd:.2f}"] = speckle_noise(images, sd, rng)
    for sev in [0.5, 1.0, 2.0]:
        conds[f"weather_snow_{sev}"] = snow(images, sev, rng)
    for sev in [0.3, 0.5, 0.8]:
        conds[f"weather_frost_{sev}"] = frost(images, sev, rng)
    for sev in [0.3, 0.5, 0.7]:
        conds[f"weather_fog_{sev}"] = fog(images, sev)
    return {name: imgs for name, imgs in conds.items() if np.abs(imgs - images).mean() > 1e-4}
