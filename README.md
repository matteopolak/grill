# Grill

Grill is an image-to-ingredients machine learning model to help you find your next meal.
It is based off of a fine-tuned version of the EfficientNet V2 model.

The [Recipe1M+](http://pic2recipe.csail.mit.edu/) dataset was used to train the model.

## Training

To train the model, run the following command:

```bash
python train.py --epochs 10 --batch_size 64
```

## Inference

To run inference on an image, run the following command:

```bash
python infer.py path/to/image.jpg
```

