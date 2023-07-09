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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    # global model
    try:
        from modules import shared, sd_hijack
        if shared.sd_model is not model:
            unload_model()
            shared.sd_model = model
            sd_hijack.model_hijack.hijack(model)
            print("Loaded default model")
    except:
        print("Failed to hijack model.")


def load_model_by_url(url, list_models=None, load_models=None):
    # global list_models, load_model
    import webui.modules.sd_models
    import hashlib

    hash_object = hashlib.md5(url.encode())
    md5_hash = hash_object.hexdigest()

    from download_checkpoint import download
    download(url, md5_hash)

    webui.modules.sd_models.list_models = list_models
    webui.modules.sd_models.load_model = load_models

    webui.modules.sd_models.list_models()

    for m in webui.modules.sd_models.checkpoints_list.values():
        if md5_hash in m.name:
            load_model(m)
            break

    webui.modules.sd_models.list_models = noop
    webui.modules.sd_models.load_model = noop


@app.init
def init():

    import modules.sd_models

    modules.sd_models.list_models()
    list_models = modules.sd_models.list_models
    modules.sd_models.list_models = noop

    model = modules.sd_models.load_model()
    load_model = modules.sd_models.load_model

    modules.sd_models.list_models = noop

    # webui.initialize()
    modules.script_callbacks.app_started_callback(None, app_fastapi)
    register_model(model=model)

    context = {
        "model": model
    }

    return context

@app.handler(route="/txt2img")
def handler(context: dict, request: Request) -> Response:
    body = request.json.get("body")
    # model_input = json.loads(body)

    params = body["params"]
    model_parameter = reqmodels.StableDiffusionTxt2ImgProcessingAPI(**params)

    # webui.initialize()
    modules.script_callbacks.app_started_callback(None, app_fastapi)
    text_to_image = Api(app_fastapi, queue_lock)
    response = text_to_image.text2imgapi(model_parameter)

    return Response(
        json={"output": response.images[0]},
        status=200
    )

@app.handler()
def default(context: dict, request: Request) -> Response:
    return Response(json={"output": "success"}, status=200)


if __name__ == "__main__":
    app.serve()
