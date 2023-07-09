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

### Client Implementation in Python

```Python
import banana_dev as client

my_model = client.Client(
    api_key="[API_KEY]",
    model_key="[MODEL_KEY]",
    url="[MODEL_URL]",
)

inputs = {
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

# Call your model's inference endpoint on Banana.
# If you have set up your Potassium app with a
# non-default endpoint, change the first
# method argument ("/")to specify a
# different route.
result, meta = my_model.call("/text2img", inputs)

print(result)
```