"""
mosaic_creator.py

Author: Tobias Weber
Created: 2023-12-01
Description: A script to generate photo mosaics from a source image and a set of tiles.
Version: 1.0
License: MIT

Usage:
    P = Image.open(target_path)
    T = ...  # Load tiles as list of PIL Images
    params = {
        'resolution': tile_res,
        'granularity': r,  # number of subdivisions
        'tile_count': n,  # number of tiles in target image
        'repetitions': 1,
        'crop_count': 1
    }
    builder = MosaicBuilder(photo=P, tile_images=T, params=params)
    generated = builder.build()
    mosaic = generated.mosaic
"""

import time
from PIL import Image, ImageOps
import numpy as np
from scipy import optimize
import math
import random


class Mosaic:
    def __init__(self, photo: Image, mosaic: Image, shape: tuple[int, int], assignment: np.ndarray):
        self.photo = photo
        self.mosaic: Image = mosaic
        self.shape = shape
        self.assignment = assignment

    def get_score(self):
        if self.photo.size[0] > self.mosaic.size[0]:
            a = self.photo.resize(self.mosaic.size, resample=Image.Resampling.BILINEAR)
            a = np.asarray(a).astype('int16')
            b = np.asarray(self.mosaic).astype('int16')
        else:
            a = np.asarray(self.photo).astype('int16')
            b = self.mosaic.resize(self.photo.size, resample=Image.Resampling.BILINEAR)
            b = np.asarray(b).astype('int16')
        norms = np.linalg.norm(a - b, axis=-1)
        return norms.mean()

    def get_processed_mosaic(self, overlay_opacity):
        if 1.0 >= overlay_opacity > 0.0:
            return blend_images(self.mosaic, self.photo, overlay_opacity)
        return self.mosaic


class MosaicBuilder:
    def __init__(self, photo=None, tile_images=(), params=None):
        self.tile_images = tile_images
        self.photo: Image = photo
        if params:
            self.set_tile_res(params['resolution'])
            self.set_granularity(params['granularity'])
            self.set_repetitions(params['repetitions'])
            self.set_crop_count(params['crop_count'])
            self.set_tile_count(params['tile_count'])
        else:
            self.shape = (1, 1)  # (vertical no. of tiles, horizontal no. of tiles)
            self.tile_res: int = 64
            self.granularity = 1
            self.repetitions = 1
            self.crop_count = 1
            self.tile_count = 0

    def set_tile_images(self, images):
        self.tile_images = images
        return self

    def set_photo(self, photo: Image):
        self.photo = photo
        return self

    def set_tile_count(self, max_tiles: int):
        if max_tiles < 1:
            raise ValueError('Invalid tile count.')
        self.tile_count = max_tiles
        return self

    def set_tile_res(self, tile_res: int):
        self.tile_res = tile_res
        return self

    def set_granularity(self, granularity: int):
        self.granularity = granularity
        return self

    def set_repetitions(self, max_repetitions: int):
        self.repetitions = max_repetitions
        return self

    def set_crop_count(self, crop_count: int):
        self.crop_count = crop_count
        return self

    def build(self, progress_callback=None) -> Mosaic:
        if not all([self.tile_images, self.photo]):
            raise ValueError("Not all required attributes have been specified. Cannot build the mosaic.")

        if self.tile_count > 0:
            self.shape = shape_from_count(self.photo, self.tile_count)
        else:
            w, h = self.photo.size
            self.shape = (int(h // self.tile_res), int(w // self.tile_res))

        mosaic, assignment = self._build_mosaic(progress_callback)
        return Mosaic(self.photo, mosaic, self.shape, assignment)

    def _build_mosaic(self, progress_callback=None):
        progress_callback(0) if progress_callback else None
        tiles = self._get_tiles()
        tile_vals = self._get_tile_vals(tiles)
        progress_callback(1) if progress_callback else None
        C, C_choice = self._get_best_C(tile_vals)
        progress_callback(2) if progress_callback else None
        col_ind = self._get_assignment(C, C_choice)
        progress_callback(3) if progress_callback else None
        mosaic = self._get_mosaic(tiles, col_ind)
        progress_callback(4) if progress_callback else None
        return mosaic, col_ind

    def _get_tiles(self):
        size = (self.tile_res, self.tile_res)
        if self.crop_count > 1:
            tiles = []
            positions = np.linspace(0.0, 1.0, self.crop_count)
            for img in self.tile_images:
                tiles += [ImageOps.fit(img, size, Image.LANCZOS, centering=(pos, pos)) for pos in positions]
            return tiles
        else:
            return [ImageOps.fit(img, size, Image.LANCZOS, centering=(0.5, 0.5)) for img in self.tile_images]

    def _get_tile_vals(self, tiles) -> np.ndarray:
        size = (self.granularity, self.granularity)
        tile_vals = np.zeros((len(tiles), self.granularity, self.granularity, 3))
        for i, img in enumerate(tiles):
            tile_vals[i] = np.asarray(img.resize(size, resample=Image.Resampling.BILINEAR))
        return tile_vals
    
    def _get_C(self, tile_vals):
        g = self.granularity

        if self.shape[0] * g == self.photo.size[1] and self.shape[1] * g == self.photo.size[0]:
            print('Detected maximum granularity')
            photo_arr = np.asarray(self.photo)
        else:
            photo_arr = np.asarray(self.photo.resize(
                (self.shape[1] * g, self.shape[0] * g),
                resample=Image.Resampling.BILINEAR))

        n = self.shape[0] * self.shape[1]
        rows, cols = self.shape

        C = np.zeros((n, len(tile_vals)))
        photo_crops = np.zeros((n, g * g * 3))

        for i in range(n):
            row = i // cols
            col = i % cols
            photo_crops[i] = photo_arr[row * g:row * g + g, col * g:col * g + g, :].flatten()
        
        # we think about feature vectors, not images, so we flatten each one
        tile_vals = tile_vals.reshape(tile_vals.shape[0], -1)

        # This is the slow part:
        photo_crops = photo_crops.astype('int64')  # 16 enough for rgb

        for i in range(n):
            diffs = tile_vals - photo_crops[i]
            C[i] = np.linalg.norm(diffs, axis=-1)
        return C

    def _get_best_C(self, tile_vals):
        if self.crop_count <= 1:
            return self._get_C(tile_vals), None

        n = self.shape[0] * self.shape[1]
        Cs = np.zeros((self.crop_count, n, len(self.tile_images)))

        for i in range(self.crop_count):
            Cs[i] = self._get_C(tile_vals[i::self.crop_count])

        return np.min(Cs, axis=0), np.argmin(Cs, axis=0)

    def _get_assignment(self, C, C_choice):
        C = np.copy(C)
        n_images = C.shape[1]
        n = C.shape[0]
        reps = self.repetitions
        if n_images * reps < n:
            print(f'Not enough images ({n_images * reps} / {n}).')
            reps = math.ceil(n / n_images)
            print(f'Increasing repetitions to {reps}.')
        if reps > 1:
            C = np.tile(C, (1, reps))

        row_ind, col_ind = optimize.linear_sum_assignment(C)
        col_ind %= n_images
        if C_choice is not None:
            col_ind = col_ind * self.crop_count + C_choice[row_ind, col_ind]

        return col_ind

    def _get_mosaic(self, tiles, col_ind):
        canvas = Image.new('RGB', (self.tile_res * self.shape[1], self.tile_res * self.shape[0]))

        for i in range(self.shape[1]):
            for j in range(self.shape[0]):
                img = tiles[col_ind[j * self.shape[1] + i]]
                x, y = i * self.tile_res, j * self.tile_res
                canvas.paste(img, (x, y))
        return canvas
    

class MosaicTimer:
    def __init__(self):
        self.timings = [0, 0, 0, 0, 0]
        self.completed = False

    def measure(self, eventId):
        if self.completed:
            raise Exception('Measurement is already completed')
        if eventId > 0 and self.timings[eventId - 1] == 0:
            raise Exception('Wrong order of measurements')
        
        t = int(round(time.time() * 1000))
        self.timings[eventId] = t
        if eventId == len(self.timings) - 1:
            self.completed = True

    def get_total_time_ms(self):
        return self.timings[-1] - self.timings[0]
    
    def get_delta(self, startId, endId):
        return self.timings[endId] - self.timings[startId]
    
    def get_C_time_ms(self):
        '''
        Return the time in ms it took to calculate the cost matrix.
        '''
        return self.get_delta(1, 2)
    
    def get_LAP_time_ms(self):
        '''
        Return the time in ms it took to solve the assignment problem.
        '''
        return self.get_delta(2, 3)


def shape_from_count(img: Image, n: int):
    w, h = img.size
    no_px = h * w
    px_per_tile = no_px / n
    tile_side = math.sqrt(px_per_tile)
    return int(h // tile_side), int(w // tile_side)


def blend_images(img1, img2, opacity):
    resized2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    return Image.blend(img1, resized2, opacity)
