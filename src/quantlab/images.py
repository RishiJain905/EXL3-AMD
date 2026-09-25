"""Bounded local image inputs. HTTP images never fetch URLs or read paths."""
import base64
import binascii
from io import BytesIO
from pathlib import Path

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8 * 1024**2
MAX_SOURCE_PIXELS = 16 * 1024**2
IMAGE_BODY_BYTES = 12 * 1024**2
DEFAULT_IMAGE_PIXELS = 262144


class ImageInputError(ValueError):
    """Safe, fixed image-input diagnostics suitable for the local API."""


def image_url_bytes(value):
    if not isinstance(value, dict) or set(value) - {'url', 'detail'}:
        raise ImageInputError('image_url must contain url and optional detail')
    if value.get('detail', 'auto') != 'auto':
        raise ImageInputError('Only image detail=auto is supported; use --image-max-pixels')
    url = value.get('url')
    if not isinstance(url, str):
        raise ImageInputError('Image URL must be a base64 PNG or JPEG data URL')
    prefix, separator, encoded = url.partition(',')
    if not separator or prefix not in ('data:image/png;base64', 'data:image/jpeg;base64'):
        raise ImageInputError('Images require base64 PNG or JPEG data URLs; remote URLs and paths are unsupported')
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageInputError('Image exceeds the 8 MiB compressed limit')
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageInputError('Invalid base64 image data') from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError('Image must contain 1 byte to 8 MiB')
    expected = 'PNG' if prefix == 'data:image/png;base64' else 'JPEG'
    return data, expected


def decode_image(data, expected_format=None):
    """Check headers before allocating pixels; detach RGB from the byte stream."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError('Image must contain 1 byte to 8 MiB')
    try:
        with BytesIO(data) as stream, Image.open(stream, formats=('PNG', 'JPEG')) as im:
            if expected_format is not None and im.format != expected_format:
                raise ImageInputError('Image data does not match its declared MIME type')
            if im.width * im.height > MAX_SOURCE_PIXELS:
                raise ImageInputError('Image exceeds the 16 megapixel source limit')
            if getattr(im, 'n_frames', 1) != 1:
                raise ImageInputError('Animated images are unsupported')
            with ImageOps.exif_transpose(im) as oriented:
                # Match the source processor's white background for alpha,
                # including palette transparency and grayscale-plus-alpha.
                if oriented.mode == 'RGB':
                    result = oriented.copy()
                else:
                    with oriented.convert('RGBA') as rgba, Image.new('RGBA', oriented.size, 'white') as background:
                        background.alpha_composite(rgba)
                        result = background.convert('RGB')
                result.load()
                return result
    except ImageInputError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ImageInputError('Invalid or unsupported PNG/JPEG image') from None


def local_image_url(path):
    """CLI-only path reader; never called with an HTTP field."""
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_IMAGE_BYTES + 1)
    with decode_image(data):
        pass
    mime = 'image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg'
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


def validate_vision_options(args):
    pixels = getattr(args, 'image_max_pixels', DEFAULT_IMAGE_PIXELS)
    if type(pixels) is not int or not 65536 <= pixels <= 1048576 or pixels % 1024:
        raise ValueError('--image-max-pixels must be a multiple of 1024 in [65536,1048576]')
    if getattr(args, 'mmproj', 'off') != 'on':
        if getattr(args, 'image', None):
            raise ValueError('--image requires --mmproj on')
        return
    if getattr(args, 'decode_fusions', 'off') != 'off' or any(getattr(args, name, False) for name in (
            'native_attention', 'draft_step_graph', 'gpu_embedding', 'gpu_draft', 'shortlist_groups')):
        raise ValueError('Vision requires --decode-fusions off and no native attention, draft graph, GPU embedding/draft or shortlist')
    if getattr(args, 'cache_mtp', 'off') != 'off' or getattr(args, 'prefix_cache', 'off') != 'off':
        raise ValueError('Vision currently requires MTP projection and prefix caches off')
    mode = getattr(args, 'mode', 'serve')
    if mode not in ('generate', 'serve'):
        raise ValueError('--mmproj on currently supports generate and serve modes')
    if getattr(args, 'expected_results', None):
        raise ValueError('--expected-results is unavailable for vision generation')
    if getattr(args, 'image', None) and mode != 'generate':
        raise ValueError('--image applies only to generate mode')
    if len(getattr(args, 'image', None) or []) > MAX_IMAGES:
        raise ValueError('At most four images are supported per request')
