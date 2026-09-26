"""Optional Qwen3.5 vision residency and explicit embedding/template binding."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from quantlab.images import ImageInputError, MAX_IMAGES, decode_image, image_url_bytes


def configure_vision(config, candidate, max_pixels):
    """Enable only the supported component on the existing mapped text config."""
    from exllamav3.architecture.qwen3_vl import (
        Qwen3VLVisionModel, read_qwen3_vl_vision_config, read_qwen3_vl_pp_config)
    raw = json.loads((Path(candidate) / 'config.json').read_text())
    if raw.get('model_type') != 'qwen3_5' or raw.get('vision_config', {}).get('deepstack_visual_indexes') != []:
        raise ValueError('--mmproj on currently supports dense Qwen3.5 with no deepstack layers')
    pp_path = Path(candidate) / 'preprocessor_config.json'
    pp = json.loads(pp_path.read_text())
    if any(pp.get(key, True) is not True for key in ('do_convert_rgb', 'do_normalize', 'do_rescale', 'do_resize')):
        raise ValueError('Unsupported disabled image preprocessing operation')
    if pp.get('resample', 3) != 3 or pp.get('rescale_factor', 1 / 255) != 1 / 255:
        raise ValueError('Unsupported image resampling or rescale factor')
    # These names share the same saved numerical settings. The vendored
    # implementation executes its own explicit patchify/normalize pipeline.
    if pp.get('image_processor_type') not in ('Qwen2VLImageProcessor', 'Qwen2VLImageProcessorFast'):
        raise ValueError('Unsupported Qwen3.5 image processor')
    normalized = dict(pp, image_processor_type='Qwen2VLImageProcessorFast')
    config.vision = read_qwen3_vl_vision_config(raw['vision_config'])
    config.vision_pp = read_qwen3_vl_pp_config(normalized)
    v, prep = config.vision, config.vision_pp
    if (v.patch_size, v.temporal_patch_size, v.spatial_merge_size) != (prep.patch_size, prep.temporal_patch_size, prep.merge_size):
        raise ValueError('Vision and preprocessor patch geometry disagree')
    if not prep.min_pixels <= max_pixels <= prep.max_pixels:
        raise ValueError('Image pixel cap is outside the model processor limits')
    prep.max_pixels = max_pixels  # per-process limit, never a model-file edit
    config.vision_start_token_id = raw['vision_start_token_id']
    config.vision_end_token_id = raw['vision_end_token_id']
    config.image_token_id = raw['image_token_id']
    config.model_classes['vision'] = Qwen3VLVisionModel
    tensors = config.stc.list_tensors(prefix='model.visual', only_serializable=True)
    if not tensors:
        raise ValueError('The model package has no vision tensors')
    dtypes = sorted({item['dtype'] for item in tensors.values()})
    return dict(enabled=True, component='vision', processor_type=pp['image_processor_type'],
                processor_sha256=hashlib.sha256(pp_path.read_bytes()).hexdigest(),
                min_pixels=prep.min_pixels, max_pixels=prep.max_pixels,
                patch_size=v.patch_size, merge_size=v.spatial_merge_size,
                source_precision='BF16' if dtypes == ['torch.bfloat16'] else 'mixed/other',
                source_dtypes=dtypes, source_tensor_count=len(tensors),
                source_payload_bytes=sum(item['n_bytes'] for item in tensors.values()),
                runtime_precision='inherited FP16/FP32 mixed')


def prepare_images(messages, config, *, enabled):
    """CPU-only validation and exact token-count preflight before vision GPU work.

    Replace each image with an unguessable alias rendered as ordinary text by
    the saved model template. The embedding includes its own vision boundaries;
    replacing only image_pad would incorrectly duplicate those boundaries.
    """
    from exllamav3.architecture.mm_processing.qwen2 import qwen2_smart_resize
    result, images = [], []
    try:
        for message in messages:
            content = message.get('content')
            if not isinstance(content, list):
                result.append(dict(message))
                continue
            parts = []
            for part in content:
                if part.get('type') == 'text':
                    parts.append(part['text'])
                    continue
                if part.get('type') != 'image_url' or message['role'] != 'user':
                    raise ImageInputError('Only user image_url content is supported')
                if not enabled:
                    raise ImageInputError('Images are disabled; restart with --mmproj on')
                if len(images) >= MAX_IMAGES:
                    raise ImageInputError('At most four images are supported per request')
                data, mime = image_url_bytes(part['image_url'])
                im = decode_image(data, mime)
                try:
                    v, pp = config.vision, config.vision_pp
                    size = qwen2_smart_resize(im.size, v.patch_size * v.spatial_merge_size,
                                             pp.min_pixels, pp.max_pixels)
                    if min(size) < v.patch_size * v.spatial_merge_size or size[0] * size[1] > pp.max_pixels:
                        raise ImageInputError('Image aspect ratio cannot fit the configured pixel limit')
                    count = size[0] * size[1] // (v.patch_size * v.spatial_merge_size)**2
                    alias = '<quantlab_image_' + uuid4().hex + '>'
                    images.append(dict(image=im, alias=alias, tokens=count, size=size,
                                       sha256=hashlib.sha256(data).hexdigest()))
                    parts.append(alias)
                except BaseException:
                    im.close()
                    raise
            result.append(dict(message, content=''.join(parts)))
        return result, images
    except BaseException:
        close_images(images)
        raise


def provisional_image_prompt(rendered, images, tokenizer, config):
    for item in images:
        if rendered.count(item['alias']) != 1:
            raise ImageInputError('Chat template must preserve each image exactly once')
        replacement = (tokenizer.id_to_piece[config.vision_start_token_id]
                       + tokenizer.id_to_piece[config.image_token_id] * item['tokens']
                       + tokenizer.id_to_piece[config.vision_end_token_id])
        rendered = rendered.replace(item['alias'], replacement)
    return rendered


def close_images(images):
    for item in images:
        item['image'].close()
