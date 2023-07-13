In the following wiki page is describe different methods to visualize the data captured and annotated using the OSCAR datagen tools.

## Build FiftyOne Docker image

```bash
git clone https://github.com/voxel51/fiftyone.git && cd fiftyone
make docker
```

## Start FiftyOne container

```bash
SHARED_DIR=/path/to/data
docker run -v ${SHARED_DIR}:/fiftyone -p 5151:5151 -it voxel51/fiftyone
```

## Visualize images

* Load the images:
```python
import fiftyone as fo

dataset= fo.Dataset(name="dataset_name")
dataset.add_images_patt("/path/to/images/*.png")
```

## Visualize images with its annotations

* Load the images with annotation:
```python
import fiftyone as fo

dataset= fo.Dataset(name="dataset_name")
dataset.add_dir(dataset_dir="/path/to/dataset", labels_path="/path/to/annotations.json", dataset_type=fo.types.COCODetectionDataset, data_path="rgb")
```

## Start FiftyOne web application

```python
import fiftyone as fo

session = fo.launch_app()
```

The above code will start the application at `<SERVER_IP_ADDRESS>:5151`.