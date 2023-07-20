### Banana Implementation

This repository is the implementation of [stable-diffusion-webui](!https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.3.1/modules/api/api.py)

### Endpoints

At the moment only one endpoint is implemented.

- `/txt2img` is the implementation of stable-diffusion-webui `/sdapi/v1/txt2img` api.

    Json Input
    ```JSON
    {
        "body": {
            "params": {
                "prompt": "CAT",
                "negative_prompt": "low quality",
                "steps": 25,
                "sampler_name": "Euler a",
                "cfg_scale": 7.5,
                "seed": 42,
                "batch_size": 1,
                "n_iter": 1,
                "width": 400,
                "height": 400,
                "tiling": 'false',
            }
        }
    }
    ```

    OUTPUT:
    ```JSON
    {
        "output": "<base64 image>"
    }
    ```

- `img2img` is the implementation of `/sdapi/v1/img2img` api

    JSON INPUT
    ```JSON
    img2img_inputs = {
        "params": {
            "prompt": "blue banana",
            "negative_prompt": "cartoonish, low quality",
            "steps": 25,
            "sampler_name": "Euler a",
            "cfg_scale": 7.5,
            "denoising_strength": 0.7,
            "seed": 42,
            "batch_size": 1,
            "n_iter": 1,
            "width": 768,
            "height": 768,
            "tiling": "false",
            "init_images": [
                "b64images"
            ]
        }
    }
    
    ```

### Client Implementation in Python

```Python
import banana_dev as client

my_model = client.Client(
    api_key="[API_KEY]",
    model_key="[MODEL_KEY]",
    url="[MODEL_URL]",
)

txt2img_inputs = {
    "params": {
        "prompt": "Nurse in scrub dress with her hair tied",
        "negative_prompt": "low quality",
        "steps": 25,
        "sampler_name": "Euler a",
        "cfg_scale": 7.5,
        "seed": 42,
        "batch_size": 1,
        "n_iter": 1,
        "width": 400,
        "height": 400,
        "tiling": "false",
    }
}

img2img_inputs = {
    "params": {
        "prompt": "blue banana",
        "negative_prompt": "cartoonish, low quality",
        "steps": 25,
        "sampler_name": "Euler a",
        "cfg_scale": 7.5,
        "denoising_strength": 0.7,
        "seed": 42,
        "batch_size": 1,
        "n_iter": 1,
        "width": 768,
        "height": 768,
        "tiling": "false",
        "init_images": [
            "b64images"
        ]
    }
}



txt2img_result, meta = my_model.call("/txt2img", txt2img_inputs)
img2img_result, meta = my_model.call("/img2img", img2img_inputs)

print(txt2img_inputs, img2img_result)
```