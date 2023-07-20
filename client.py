import banana_dev as client

my_model = client.Client(
    api_key="[API_KEY]",
    model_key="[API_KEY]",
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
