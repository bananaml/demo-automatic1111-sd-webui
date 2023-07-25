from potassium import Potassium, Request, Response
import threading
from modules import safe
from modules.api.api import Api
import modules
import modules.api.models as reqmodels
import webui
import torch
from fastapi import FastAPI

app = Potassium("automatic1111")
app_fastapi = FastAPI()

queue_lock = threading.Lock()

torch.load = safe.unsafe_torch_load

list_models = None
load_model = None

def noop(*args, **kwargs):
    pass

def unload_model():
    from modules import shared, sd_hijack, devices
    import gc
    if shared.sd_model:
        sd_hijack.model = None
        gc.collect()
        devices.torch_gc()

def register_model(model=None):
    try:
        from modules import shared, sd_hijack
        if shared.sd_model is not model:
            unload_model()
            shared.sd_model = model
            sd_hijack.model_hijack.hijack(model)
            print("Loaded default model")
    except:
        print("Failed to hijack model.")


@app.init
def init():

    import modules.sd_models

    modules.sd_models.list_models()
    list_models = modules.sd_models.list_models
    modules.sd_models.list_models = noop

    model = modules.sd_models.load_model()
    load_model = modules.sd_models.load_model

    modules.sd_models.list_models = noop

    modules.script_callbacks.app_started_callback(None, app_fastapi)
    register_model(model=model)
    
    return {}

@app.handler(route="/txt2img")
def handler(context: dict, request: Request) -> Response:
    params = request.json.get("params")

    if 'width' not in params:
        params['width'] = 768
    if 'height' not in params:
        params['height'] = 768

    model_parameter = reqmodels.StableDiffusionTxt2ImgProcessingAPI(**params)

    # webui.initialize()
    modules.script_callbacks.app_started_callback(None, app_fastapi)
    text_to_image = Api(app_fastapi, queue_lock)
    response = text_to_image.text2imgapi(model_parameter)

    return Response(
        json={"output": response.images[0]},
        status=200
    )

@app.handler(route="/img2img")
def imghandler(context: dict, request: Request) -> Response:
    params = request.json.get("params")

    if 'width' not in params:
        params['width'] = 768
    if 'height' not in params:
        params['height'] = 768

    model_parameter = reqmodels.StableDiffusionImg2ImgProcessingAPI(**params)
    modules.script_callbacks.app_started_callback(None, app_fastapi)
    image_to_image = Api(app_fastapi, queue_lock)
    response = image_to_image.img2imgapi(model_parameter)
    imageb64 = response.images
    
    return Response(
        json={
            "output": imageb64[0]},
            status=200
    )


@app.handler()
def default(context: dict, request: Request) -> Response:
    return Response(json={"output": "success"}, status=200)


if __name__ == "__main__":
    app.serve()
