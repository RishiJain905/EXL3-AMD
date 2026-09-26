"""Exercise an owned, bounded loopback server; never contacts a remote host."""
import asyncio
import json
from pathlib import Path
import time
import tomllib
import urllib.error
import urllib.request

from serve_exl3 import Engine, parser, reserve_socket
from quantlab.images import local_image_url


def main():
    p = parser()
    p.add_argument('--fixtures', type=Path, required=True)
    args = p.parse_args()
    config = tomllib.loads(args.config.read_text())
    if not all(config['execution'].get(k) is True for k in ('allow_local_inference','allow_backend_probes')):
        p.error('Both local execution permissions required')
    listener = reserve_socket('127.0.0.1', args.port)
    engine = None
    try:
        engine = Engine(args)
        from quantlab.server import create_app
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(create_app(engine,request_timeout=120),
            host='127.0.0.1',port=args.port,log_level='warning'))
        observations = []
        def request(body=None, path='/v1/chat/completions'):
            start = time.monotonic()
            req = urllib.request.Request('http://127.0.0.1:'+str(args.port)+path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type':'application/json'})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(req,timeout=130) as response:
                    return dict(status=response.status,body=response.read().decode(),seconds=time.monotonic()-start)
            except urllib.error.HTTPError as exc:
                return dict(status=exc.code,body=exc.read().decode(),seconds=time.monotonic()-start)
        async def run():
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                for _ in range(100):
                    if server.started: break
                    if task.done(): await task
                    await asyncio.sleep(.05)
                if not server.started: raise RuntimeError('Owned HTTP server did not start')
                async def check(name,body,status=200,path='/v1/chat/completions'):
                    observed = await asyncio.to_thread(request,body,path)
                    observations.append(dict(name=name,**observed))
                    (args.output/'http.partial.json').write_text(json.dumps(observations,indent=2))
                    if observed['status'] != status:
                        raise RuntimeError(f'{name}: expected HTTP {status}, got {observed["status"]}')
                await check('health-start',None,path='/health')
                body=dict(messages=[dict(role='user',content='Write exactly: ready')],max_tokens=8)
                await check('text-before',body)
                for i, image in enumerate(('red.png','blue.png','green-circle.png','yellow-square.png')):
                    parts=[dict(type='image_url',image_url=dict(url=local_image_url(args.fixtures.parent/image))),
                           dict(type='text',text='Name the main color and shape in the image.')]
                    image_body=dict(messages=[dict(role='user',content=parts)],max_tokens=48,stream=i==3)
                    if i==3: image_body['stream_options']=dict(include_usage=True)
                    await check('image-'+image,image_body,200 if engine.vision_enabled else 400)
                invalid=dict(messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,YmFk'))])],max_tokens=8)
                await check('invalid-image',invalid,400)
                await check('context-rejected',dict(body,max_tokens=8192),400)
                await check('text-after',dict(body,stream=True,stream_options=dict(include_usage=True)))
                await check('health-end',None,path='/health')
            finally:
                server.should_exit = True
                await asyncio.wait_for(task,15)
        with engine.torch.inference_mode():
            asyncio.run(run())
        (args.output/'result.json').write_text(json.dumps(dict(status='completed',observations=observations,
            vision=engine.vision_status,mtp_depth=engine.depth),indent=2))
    finally:
        if engine is not None: engine.shutdown()
        listener.close()


if __name__ == '__main__':
    main()
