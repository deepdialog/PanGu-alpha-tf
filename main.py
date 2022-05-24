
import traceback
from fastapi import FastAPI
from fastapi import Request
from infer import generate


app = FastAPI()


@app.post('/api/generate')
async def api_generate(request: Request):
    """
    curl -XPOST http://localhost:8000/api/generate \
        -H 'Content-Type: applicaton/json' \
        -d '{"text": "你好啊"}'
    
    curl -XPOST http://localhost:8000/api/generate \
        -H 'Content-Type: applicaton/json' \
        -d '{"text": "机器助理是一个非常聪明的，智能的机器人，它可以跟你聊天。\n用户：你是谁啊？\n机器助理：我是机器人，我来自deepdialog，你好。\n用户：你多大了？\n机器助理：", "additional_eod": ["\n"]}'
    
    """
    data = await request.json()
    if 'text' not in data or not isinstance(data['text'], str):
        return {
            'ok': False,
            'error': 'Invalid text in post data',
        }
    if 'max_len' in data:
        max_len = data.get('max_len', 50)
        if max_len >= 1000:
            return {
                'ok': False,
                'error': 'Invalid max_len',
            }

    try:
        ret = generate(
            text=data['text'],
            max_len = data.get('max_len', 50),
            temperature = data.get('temperature', 1.0),
            top_p = data.get('top_p', 0.95),
            top_k = data.get('top_k', 50),
            ban = data.get('ban', []),
            eod = data.get('eod', None),
            additional_eod = data.get('additional_eod', [])
        )
        return {
            'ok': True,
            'text': ret,
        }
    except Exception:
        return {
            'ok': False,
            'error': traceback.format_exc(),
        }


@app.get('/')
async def hello():
    return {
        'hello': 'world',
    }
