"""HTTP admission and forwarding; image tensor execution has separate GPU tests."""
import json
import unittest
from quantlab.images import ImageInputError
from quantlab.server import _Invalid, _check_messages
from quantlab.server import create_app
from quantlab.images import IMAGE_BODY_BYTES
from test_server import FakeEngine, Harness, make_scope, run_app, status_of, json_of

PART = dict(type='image_url', image_url=dict(url='data:image/png;base64,AA=='))


class ImageRequestTests(unittest.TestCase):
    def test_off_rejects_image_before_engine(self):
        with self.assertRaisesRegex(_Invalid, '--mmproj on'):
            _check_messages(dict(messages=[dict(role='user',content=[PART])]))

    def test_order_and_text_are_preserved(self):
        parts = [dict(type='text',text='before'), PART, dict(type='text',text='after')]
        messages = _check_messages(dict(messages=[dict(role='user',content=parts)]),allow_images=True)
        self.assertEqual(messages[0]['content'],parts)
        self.assertEqual(_check_messages(dict(messages=[dict(role='user',content=parts[::2])]),allow_images=True)[0]['content'],'beforeafter')

    def test_only_user_images_and_max_four(self):
        for role,parts in [('system',[PART]),('assistant',[PART]),('user',[PART]*5)]:
            with self.subTest(role=role), self.assertRaises(_Invalid):
                _check_messages(dict(messages=[dict(role=role,content=parts)]),allow_images=True)

    def test_external_urls_rejected_and_never_fetched(self):
        with self.assertRaisesRegex(_Invalid,'remote URLs'):
            _check_messages(dict(messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='http://127.0.0.1/private'))])]),allow_images=True)

    def test_unrecognized_fields_do_not_disappear(self):
        for part in (dict(PART,foo=True),dict(type='image_url',image_url=dict(PART['image_url'],detail='high'))):
            with self.assertRaises(_Invalid):
                _check_messages(dict(messages=[dict(role='user',content=[part])]),allow_images=True)


class ImageHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_vision_body_limit_is_applied_before_admission(self):
        engine=FakeEngine();engine.vision_enabled=True
        scope=make_scope('/v1/chat/completions',body=None,
                         headers=[(b'content-length',str(IMAGE_BODY_BYTES+1).encode())])
        harness=Harness(b'')
        await run_app(create_app(engine),scope,harness)
        self.assertEqual(status_of(harness),413)
        self.assertIn('12 MiB',json_of(harness)['error']['message'])
        self.assertEqual(engine.prepare_calls,[])

    async def test_decode_error_is_clear_and_does_not_start_generation(self):
        engine=FakeEngine();engine.vision_enabled=True
        engine.prepare_error=ImageInputError('Invalid or unsupported PNG/JPEG image')
        body=json.dumps(dict(messages=[dict(role='user',content=[PART])])).encode()
        harness=Harness(body)
        await run_app(create_app(engine),make_scope('/v1/chat/completions',body=body),harness)
        self.assertEqual(status_of(harness),400)
        self.assertIn('PNG/JPEG',json_of(harness)['error']['message'])
        self.assertEqual(engine.generate_calls,[])
