import json
import requests
import hmac
import hashlib
from collections import OrderedDict
import time
import base64
from PIL import Image
from io import BytesIO
def sign(params, body, app_id, secret_key):
    # 1. 构建认证字符串前缀，格式为 bot-auth-v1/{appId}/{timestamp}, timestamp为时间戳，精确到毫秒，用以验证请求是否失效
    auth_string_prefix = f"bot-auth-v1/{app_id}/{int(time.time() * 1000)}/"
    sb = [auth_string_prefix]
    # 2. 构建url参数字符串，按照参数名字典序升序排列
    if params:
        ordered_params = OrderedDict(sorted(params.items()))
        sb.extend(["{}={}&".format(k, v) for k, v in ordered_params.items()])
    # 3. 拼接签名原文字符串
    sign_str = "".join(sb) + body
    # 4. hmac_sha_256算法签名
    signature = hmac.new(
        secret_key.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    # 5. 拼接认证字符串
    return auth_string_prefix + signature


def get_base64(img):
    # 打开图片
    # img = Image.open(img_path).convert("RGB")
    # 创建一个内存缓冲区
    img_buffer = BytesIO()
    # 将图片保存为PNG格式到内存缓冲区
    img.save(img_buffer, format='PNG')
    # 获取内存缓冲区中的图片数据
    img_bytes = img_buffer.getvalue()
    # 将图片数据转换为Base64编码
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    # 返回数据，格式固定为PNG
    # return f"img_base64"
    return f"data:image/png;base64,{img_base64}"


def image_generate_url(local_path):
    # 设定 URL 地址
    url1 = 'http://sz-multimodal-test.wanyol.com/image/control/uploadImage'  # 替换为你的上传接口

    # 打开要上传的图片文件
    with open(local_path, 'rb') as img:
        # 准备文件数据，'file' 是服务器端接收文件的字段名
        files = {'file': img}
        # 发送 POST 请求
        response = requests.post(url1, files=files)
        # 输出响应结果
        # print(response.status_code)
        # print(response.text)
        return json.loads(response.content)["data"]["url"]


def get_gpt4vres(img_url,instruction,model_type="gpt-image-1"):
    try:
        ak = "*******"
        sk = "*******"
        body = {
            "model": model_type, 
            "caption": instruction,
            "images":[get_base64(img_url)],
            "size":"1024x1024",
            "quality":"auto" ## auto high medium low
            }
        data = json.dumps(body)
        header = {
            "Authorization": sign(None, data, ak, sk),
            "Content-Type": "application/json",
        }
        resp = requests.request(
            "POST",
            url="https://andesgpt-gateway-cn.heytapmobi.com/image/v1/edits",
            # url="https://andesgpt-gateway-cn.heytapmobi.com/converter/openai/v1",
            headers=header,
            data=data,
        )
        # resp.close()
        # print(resp.text)
        return resp
    except Exception as e:
        print(e)
        return None

if __name__=="__main__":
    # example()
    img_url="/mnt/data/group/**/AndesDiT/EasyControl/test_imgs/1.jpg"
    instruction="每一个人都变成光头"
    resp=get_gpt4vres(img_url,instruction)
    print(resp.text)