# Introduction

This projects aims to determine the municipality of Cadiz's province which has the most number of cars outdoor.

# Methodology

As manual counting each car would be inefficient, I would make use of YOLO OBB models.

## Fine-tuning

Testing multiple YOLO OBB models I ended up realising that they were bad at recognising cars in the shadows or partially covered.

To improve this I Auto-labeled +1900 images and manually reinforce and improve the results, so after this I could fine-tune the final YOLO OBB selected model. 

# Timeline

17/09/2026 -> 19/09/2026 - Start testing different YOLO OBB model (8x, 11x, 26x) trained with DOTAv1.0 (aerial images) that fits the use case I'm looking for

21/09/2026 -> 22/09/2026 - Bulk download of +1000 satellite images from CNIG of the province of Cadiz

25/09/2026 -> 29/09/2026 - Auto-labeling and manual tweaking of +1900 images (1024x1024). Finally, I discard many images because they where many empty space (water, blurred images, black, grass)

29/09/2026 -> 30/09/2026 - Testing different training models (freezing layers, epochs, etc)

30/09/2026 -> xx/xx/xxxx - Converting the 90GB raw satellite images to a better format for web viewing without dealing with TB's of data (lol)

# Results

# Sources

* Satellite images: [CNIG](https://centrodedescargas.cnig.es/CentroDescargas/ortofotos-provisionales-pnoa) (_Centro Nacional de Información Geográfica_)
* Geospatial data [OpenStreetMap](https://www.openstreetmap.org/) and processed by [Geofabrik Gmbh](https://www.geofabrik.de)