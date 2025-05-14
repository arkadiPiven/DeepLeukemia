import pandas as pd
from PIL import Image
import random
import pickle
import os
from typing import List, Tuple
import openslide
import numpy as np
from tqdm import tqdm
from numpy.typing import ArrayLike
from utils.utils import remove_suffix, MRXS_SUFFIX, GRID_256_SUFFIX
from dataclasses import dataclass


@dataclass
class SlideNameAggregator:
    patches_count: int
    patches_grid: ArrayLike
    slide_header: openslide.OpenSlide | None


class SlideHandler:
    """
    As of the moment only supports ".mrxs" type whole slide images.

    SlidesHandler purpose is to govern whole slide images access.
    From accessing individual patches to
    """

    def __init__(self,
                 annotations_df: pd.DataFrame,
                 img_dirs: List[str],
                 grids: List[str],
                 sample_size: int = 16):
        """
        SegmentSlides initializer.
        Uses paths provided in envs file to read whole slide images, in case not provided.
        Those parameters are not mandatory, but can reduce code duplication.
        :param img_dirs: List of all involved image directories.
        :param grids: List of all involved grids directories.
        """
        # Initialize default paths
        self._total_patches_derived: int = 0
        self._total_slides: int = annotations_df.shape[0]
        self._grids_dirs: List[str] = grids.copy()
        self._img_dirs: List[str] = img_dirs.copy()
        self._sample_size: int = sample_size
        self._filename_to_slide_name_aggregator: dict[str, SlideNameAggregator] = dict()
        self._filenames_ordered: List[str] = list()

        for index, row in annotations_df.iterrows():
            slide_name: str = row["slide_name"]

            pixels: np.typing.ArrayLike = np.zeros(1)

            for path in self._grids_dirs:
                if remove_suffix(slide_name, MRXS_SUFFIX) + GRID_256_SUFFIX in os.listdir(path):
                    with open(f"{path}/{remove_suffix(slide_name, MRXS_SUFFIX) + GRID_256_SUFFIX}",
                              "rb") as file_handle:
                        pixels = np.array(pickle.load(file_handle))
                    self._total_patches_derived += len(pixels)
                    break

            # Shouldn't create it here.
            self._filename_to_slide_name_aggregator[slide_name] = SlideNameAggregator(patches_count=len(pixels),
                                                                                      patches_grid=pixels,
                                                                                      slide_header=None)
            self._filename_to_slide_name_aggregator = dict(sorted(self._filename_to_slide_name_aggregator.items()))
            self._filenames_ordered = list(self._filename_to_slide_name_aggregator.keys()).copy()

    @property
    def total_slides(self) -> int:
        return self._total_slides

    @property
    def total_patches_derived(self) -> int:
        """
        Getter for total patches derived
        :return: Integer which is the number of derived patches
        """
        return self._total_patches_derived

    @total_patches_derived.setter
    def total_patches_derived(self, value):
        """
        Setter for all the number of derived patches.
        :param value: value to set.
        """
        self._total_patches_derived = value

    def get_number_of_patches_per_slide(self, slide_name) -> int:
        """
        Getter for number of patches, provided slide_name
        """
        return self._filename_to_slide_name_aggregator[slide_name].patches_count

    def get_patch_from_slide_by_index(self, slide_name: str, idx: int, level_count: int = 2,
                                      size: Tuple[int, int] = (256, 256)) -> Image:
        """
        Return patch from a slide by using index from grid.
        """
        sna = self._filename_to_slide_name_aggregator[slide_name]
        if sna.slide_header is None:
            for path in self._img_dirs:
                if slide_name in os.listdir(path):
                    sna.slide_header = openslide.OpenSlide(f"{path}/{slide_name}")
        patch = sna.slide_header.read_region(np.flip(sna.patches_grid[idx]), level_count, size)
        return patch

    def get_random_patch_by_idx(self, idx: int, level_count: int = 2, size: Tuple[int, int] = (256, 256)) -> (
            str, Image):
        slide_idx = idx // self._sample_size
        slide_name = self._filenames_ordered[slide_idx]
        sna = self._filename_to_slide_name_aggregator[slide_name]

        if sna.slide_header is None:
            for path in self._img_dirs:
                if slide_name in os.listdir(path):
                    sna.slide_header = openslide.OpenSlide(f"{path}/{slide_name}")
                    break

        random_integer = random.randint(0, sna.patches_count - 1)
        coordinates = np.flip(sna.patches_grid[random_integer])
        patch = sna.slide_header.read_region(coordinates, level_count, size)
        return slide_name, coordinates, patch

    def get_patch_by_index(self, idx: int, level_count: int = 2, size: Tuple[int, int] = (256, 256)) -> (str, Image):
        """
        Returns patch by given index, level count and size of the patch.
        :param idx: Index integer. The index is relative to all loaded images and segmentation data. Type int
        :param level_count: At what level count to extract the patch. Type int.
        :param size: What size of patch to return. Type tuple int, int.
        :return: Tuple of the filename from which the patch was extracted and the Patch itself. Types str, Image
        """
        total_patch_count: int = 0
        patch_filename: str = str()

        patch_offset = -1

        for filename, sna in self._filename_to_slide_name_aggregator.items():
            total_patch_count += sna.patches_count
            if idx < total_patch_count:
                patch_filename = filename
                patch_offset = idx - (total_patch_count - sna.patches_count)
                break
        if patch_filename == "":
            return None  # TODO: Need to catch this outside
        sna = self._filename_to_slide_name_aggregator[patch_filename]

        if sna.slide_header is None:
            for path in self._img_dirs:
                if patch_filename in os.listdir(path):
                    sna.slide_header = openslide.OpenSlide(f"{path}/{patch_filename}")
                    break
        patch = sna.slide_header.read_region(np.flip(sna.patches_grid[patch_offset]), level_count, size)
        coordinates = np.flip(sna.patches_grid[patch_offset])
        return patch_filename, coordinates, patch

    def __del__(self):
        """
        Destructor for the class
        """
        for _, sna in self._filename_to_slide_name_aggregator.items():
            if sna.slide_header is not None:
                sna.slide_header.close()

    class OpenSlideContext:
        """
        Openslide wrapper for dealing with exceptions and errors when opening wsi.
        """

        def __init__(self, path):
            self.slide = openslide.OpenSlide(path)

        def __enter__(self):
            return self.slide

        def __exit__(self, type, value, traceback):
            self.slide.close()

    @staticmethod
    def save_image_to_dir(path: str, image: Image, filename: str = "image.png", ):
        """
        Saves a given image to path under filename.
        Create directory at path. otherwise creates one.
        :param path: str
        :param image: PIL.Image
        :param filename: str
        """
        try:
            os.makedirs(path, exist_ok=True)
            image.save(path + f"/{filename}", "PNG")
        except OSError as e:
            print(f"Couldn't save {filename} to disk with error:", e)

    def peek(self,
             image_path: str,
             output_path: str = "peek",
             save_to_disk: bool = False,
             coord: Tuple[int, int] = (0, 0),
             show_image: bool = False,
             peek_level: int = 0,
             patch_dimensions: Tuple[int, int] = (256, 256)) -> Image:
        """
        Peeks at a patch.

        :param image_path: Path to image.
        :param output_path: In case of saving to disk, path to output folder.
        :param save_to_disk: Boolean indicating if to save or not.
        :param coord: Coordinates of the patch
        :param show_image: If true, shows the image.
        :param peek_level: On what level to take the patch.
        :param patch_dimensions: Patch dimensions.
        :return: Returns the image.
        """
        # TODO: Update resource list after saving to disk.
        try:
            if not os.path.isfile(image_path):
                print(f"{image_path} doesn't exist")
                return None
        except OSError as e:
            print("OS error in checking if image exists with:", e)
            return None
        with self.OpenSlideContext(image_path) as wsi:
            patch = wsi.read_region(coord, peek_level, patch_dimensions)
            if show_image:
                try:
                    patch.show()
                except OSError as e:
                    print("An error occurred while trying to display the patch:", e)
            if save_to_disk:
                self.save_image_to_dir(path=output_path, image=patch, filename="patch.png")

        return patch


def main():
    # YOUR PATHS HERE
    train_slides_path = ""
    train_grids_path = ""
    test_slides_path = ""
    test_grids_path = ""

    cnt = 0

    for file in tqdm(os.listdir(train_slides_path)):
        if cnt == 5:
            break
        if file.endswith(MRXS_SUFFIX):
            os.makedirs("" + file.removesuffix(".mrxs"),
                        exist_ok=True)
            with open(f"{train_grids_path}/{file.removesuffix('.mrxs') + GRID_256_SUFFIX}",
                      "rb") as file_handle:
                pixels: np.ndarray = np.array(pickle.load(file_handle))
            with openslide.OpenSlide(train_slides_path + "/" + file) as slide:
                indices_list = random.sample(range(len(pixels)), 30)

                for index in indices_list:
                    coords = np.flip(pixels[index])
                    patch = slide.read_region(coords, 2, (256, 256))
                    patch.save("" + file.removesuffix(
                        "") + f"/{coords}.png", "PNG")
            cnt += 1

    cnt = 0
    for file in tqdm(os.listdir(test_slides_path)):
        if cnt == 5:
            break
        if file.endswith(MRXS_SUFFIX):
            os.makedirs("" + file.removesuffix(".mrxs"),
                        exist_ok=True)
            with open(f"{test_grids_path}/{file.removesuffix('.mrxs') + GRID_256_SUFFIX}",
                      "rb") as file_handle:
                pixels: np.ndarray = np.array(pickle.load(file_handle))
            with openslide.OpenSlide(test_slides_path + "/" + file) as slide:
                indices_list = random.sample(range(len(pixels)), 30)

                for index in indices_list:
                    coords = np.flip(pixels[index])
                    patch = slide.read_region(coords, 2, (256, 256))
                    patch.save("" + file.removesuffix(
                        ".mrxs") + f"/{coords}.png", "PNG")
            cnt += 1


if __name__ == "__main__":
    main()
