# Deep Photo Style Transfer

## Training Instructions:
1. Run base model with the following: `python worker.py -c configs/base_model.yaml`
2. Generated images will be saved every 100 epochs in `/out`.

## Image Segmentation
1. Copy the image you want segmented into `/data/seg_test`
2. cd to `/semantic-segmentation-pytorch` and run `./demo_test.sh` -- note: `.jpeg` images but have `.jpg` file extension. 
3. If the GPU runs out of memory, you'll need to crop the image yourself (~2000px seems to work)
4. Change style and content images in the content file to match the image you want to use in `/data/seg_test`
5. Run the model with `use_mask: True` in config. 