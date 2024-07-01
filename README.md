# Grill

Grill is a machine learning model that takes in an image of a recipe and outputs a list of ingredients, trained on the [Recipe1M dataset](http://pic2recipe.csail.mit.edu/).

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

