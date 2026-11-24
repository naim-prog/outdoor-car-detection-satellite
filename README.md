# Introduction

The objective of this project is to improve a Computer Vision AI model to enhance its ability to detect land vehicles using satellite imagery. Potentially identifying areas where parking spaces for vehicles are needed.

That said, the goal of this project is to expand my knowledge of AI and improve my programming skills.

# Methodology

Testing multiple YOLO OBB models (8, 11, 26) I ended up realising that they were bad at recognising cars in the shadows or partially covered.

I finally decided to take YOLO26x-OBB as the base model for this project. But it still fails to detect many cars in pretty clear conditions and good lighting.

Using X-Anylabeling software I use the base model to create a labeling base of all the vehicles in 1 image (1024px * 1024px) and manually modify misplaced labels and adding those missing. 

After labeling many images I needed to finetune the base model with my own dataset.

## Fine-tuning

The process was split between 2 phases:
1. Training 60 epochs with first 10 layers frozen with a reasonable learning rate to change the deep layers of the model based on YOLO26x-OBB model (architecture and weights).
2. Second training 40 epochs with all layers unfrozen and very small learning rate in order to slightly change the head of the model.

# Timeline

17/09/2026 -> 19/09/2026 - Start testing different YOLO OBB model (8x, 11x, 26x) trained with DOTAv1.0 (aerial images) that fits the use case I'm looking for.

21/09/2026 -> 22/09/2026 - Bulk download of +1000 satellite images from CNIG of the province of Cadiz.

25/09/2026 -> 29/09/2026 - Auto-labeling and manual tweaking of +1900 images (1024x1024). Finally, I discard many images because they where many empty space (water, blurred images, black, grass) and endup with ±1100 images.

29/09/2026 -> 30/09/2026 - Testing different training models (freezing layers, epochs, etc).

30/09/2026 -> 15/10/2026 - Execution of command `gdal_translate` to create MBTiles file.

15/10/2026 -> 21/10/2026 - Execution of command `gdaladdo` to add different zoom levels to the MBTiles file.

21/10/2026 -> 23/10/2026 - Sanitizing MBTiles file (reducing size by deleting zoom levels), conversion to PMTiles and web testing.

24/10/2026 -> 02/11/2026 - Auto-labeling and manual tweaking of +600 images (1024x1024). Endup with ±1800 images (80% train and 20% val).

03/11/2026 -> 07/11/2026 - Inference on the hole dataset with YOLO26x Base model.

07/11/2026 -> 15/11/2026 - Fine-tuning YOLO OBB model with own dataset.

15/11/2026 -> 22/11/2026 - Inference on the hole dataset with YOLO26x Finetuned model.

24/11/2026 -> 24/11/2026 - Exporting results on `test` dataset

# Results

| Model            | mAP50-95 |  mAP50    | Precision |  Recall   |
|:----------------:|:--------:|:---------:|:---------:|:---------:|
| YOLO26x-OBB base | 0.7400   | 0.8283    | 0.8704    | 0.9068    |
| Finetuned model  | 0.8556   | 0.9807    | 0.9667    | 0.9505    |

*<small>Final metrics were evaluated on a dataset of 114 images labeled using YOLO26x-OBB base model and manually adding missing labels.</small>

# Sources

* Satellite images: [CNIG](https://centrodedescargas.cnig.es/CentroDescargas/ortofotos-provisionales-pnoa) (_Centro Nacional de Información Geográfica_)
* Geospatial data [OpenStreetMap](https://www.openstreetmap.org/) and processed by [Geofabrik Gmbh](https://www.geofabrik.de)
