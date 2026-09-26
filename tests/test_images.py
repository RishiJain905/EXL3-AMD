import base64
import importlib.util
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from quantlab.images import (ImageInputError, decode_image, image_url_bytes,
                             validate_vision_options)


class ImageOptionsTests(unittest.TestCase):
    def test_default_is_off_and_an_image_requires_opt_in(self):
        validate_vision_options(SimpleNamespace())
        with self.assertRaisesRegex(ValueError, 'requires --mmproj on'):
            validate_vision_options(SimpleNamespace(image=['x.png']))

    def test_unsupported_combinations_reject(self):
        for extra in ({'prefix_cache':'on'}, {'gpu_embedding':True},
                      {'decode_fusions':'gdn'}, {'mode':'speed'}, {'image_max_pixels':16777216},
                      {'image_max_pixels':65537}, {'image':['a']*5}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_vision_options(SimpleNamespace(mmproj='on', **extra))

    def test_data_only_and_explicit_detail(self):
        for url in ('https://localhost/picture.png', 'file:///secret', '/tmp/image.png',
                    'data:image/svg+xml;base64,AAAA', 'data:image/png;base64,!'):
            with self.subTest(url=url), self.assertRaises(ImageInputError):
                image_url_bytes(dict(url=url))
        with self.assertRaisesRegex(ImageInputError, 'detail=auto'):
            image_url_bytes(dict(url='data:image/png;base64,AAAA', detail='high'))


@unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow unavailable')
class ImageDecodeTests(unittest.TestCase):
    def image_data(self, format='PNG'):
        from PIL import Image
        with Image.new('RGB',(64,32),(210,20,10)) as image, BytesIO() as buffer:
            image.save(buffer,format=format)
            return buffer.getvalue()

    def test_decoded_pixels_survive_closed_source(self):
        data = self.image_data()
        decoded, mime = image_url_bytes(dict(url='data:image/png;base64,'+base64.b64encode(data).decode()))
        with decode_image(decoded,mime) as image:
            self.assertEqual(image.size,(64,32))
            self.assertEqual(image.getpixel((20,20)),(210,20,10))

    def test_invalid_truncated_and_mime_mismatch_rejected(self):
        for data,mime in ((b'not an image','PNG'), (self.image_data()[:40],'PNG'),
                          (self.image_data('JPEG'),'PNG')):
            with self.subTest(mime=mime), self.assertRaises(ImageInputError):
                decode_image(data,mime)

    def test_source_dimension_limit_precedes_pixel_load(self):
        with patch('quantlab.images.MAX_SOURCE_PIXELS',100), self.assertRaisesRegex(ImageInputError,'source limit'):
            decode_image(self.image_data())

    def test_transparency_composites_onto_white(self):
        from PIL import Image
        with Image.new('RGBA',(32,32),(255,0,0,0)) as image, BytesIO() as buffer:
            image.save(buffer,format='PNG')
            with decode_image(buffer.getvalue()) as decoded:
                self.assertEqual(decoded.getpixel((0,0)),(255,255,255))

    def test_exif_orientation_applied(self):
        from PIL import Image
        with Image.new('RGB',(64,32),'red') as image, BytesIO() as buffer:
            exif=Image.Exif();exif[274]=6
            image.save(buffer,format='JPEG',exif=exif)
            with decode_image(buffer.getvalue()) as decoded:
                self.assertEqual(decoded.size,(32,64))


class ImageTemplateTests(unittest.TestCase):
    def test_boundaries_insert_once_and_each_image_is_required(self):
        from quantlab.methods.exl3.vision import provisional_image_prompt
        tokenizer = SimpleNamespace(id_to_piece=['<start>','<image>','<end>'])
        config = SimpleNamespace(vision_start_token_id=0,image_token_id=1,vision_end_token_id=2)
        images = [dict(alias='unique_image_alias',tokens=3)]
        self.assertEqual(provisional_image_prompt('Before unique_image_alias after',images,tokenizer,config),
                         'Before <start><image><image><image><end> after')
        for text in ('missing image', 'unique_image_alias unique_image_alias'):
            with self.assertRaises(ImageInputError):
                provisional_image_prompt(text,images,tokenizer,config)


if __name__ == '__main__':
    unittest.main()
