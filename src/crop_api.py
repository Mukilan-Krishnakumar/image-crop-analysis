"""
copyright 2021 twitter, inc.
spdx-license-identifier: apache-2.0
"""

from pathlib import path
from collections import namedtuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import rectangle
from matplotlib.collections import patchcollection
from pil import image

import shlex, subprocess
import tempfile
import logging

croprectangle = namedtuple("croprectangle", "left top width height")


def reservoir_sampling(stream, k=5):
    reservoir = []
    for i, item in enumerate(stream, start=1):
        if i <= k:
            reservoir.append(item)
            continue
        # append new element with prob k/i
        sample_prob = k / i
        should_append = np.random.rand() < sample_prob
        if should_append:
            # replace random element with new element
            rand_idx = np.random.randint(k)
            reservoir[rand_idx] = item
    return reservoir


def parse_output(output):
    output = output.splitlines()
    final_output = {
        "salient_point": [],
        "crops": [],
        "all_salient_points": [],
    }
    key = "salient_point"
    for i, line in enumerate(output):
        line = line.split()
        if len(line) in {2, 4}:
            line = [int(v) for v in line]
            if i != 0:
                key = "crops"
        elif len(line) == 3:
            key = "all_salient_points"
            line = [float(v) for v in line]
        else:
            raise runtimeerror(f"invalid line: {line}")
        final_output[key].append(line)
    return final_output


def fit_window(center: int, width: int, maxwidth: int):
    if width > maxwidth:
        raise runtimeerror("error: width cannot exceed maxwidth")

    fr: int = center - width // 2
    to: int = fr + width

    if fr < 0:
        # window too far left
        fr = 0
        to = width
    elif to > maxwidth:
        # window too far right
        to = maxwidth
        fr = to - width
    return fr, to


def generate_crop(img, x, y, targetratio):
    (
        imageheight,
        imagewidth,
    ) = img.shape[:2]
    imageratio: float = (imageheight) / imagewidth

    if targetratio < imageratio:
        # squeeze vertically
        window = fit_window(y, np.round(targetratio * imagewidth), imageheight)
        top = window[0]
        height = max(window[1] - window[0], 1)
        left = 0
        width = imagewidth
    else:
        # squeeze horizontally
        window = fit_window(x, np.round(imageheight / targetratio), imagewidth)
        top = 0
        height = imageheight
        left = window[0]
        width = max(window[1] - window[0], 1)

    rect = croprectangle(left, top, width, height)
    return rect


def is_symmetric(
    image: np.ndarray, threshold: float = 25.0, percentile: int = 95, size: int = 10
) -> bool:
    if percentile > 100:
        raise runtimeerror("error: percentile must be between 0 and 100")
        return false

    # downsample image to a very small size
    mode = none
    if image.shape[-1] == 4:
        # image is rgba
        mode = "rgba"
    imageresized = np.asarray(
        image.fromarray(image, mode=mode).resize((size, size), image.antialias)
    ).astype(int)
    imageresizedflipped = np.flip(imageresized, 1)

    # calculate absolute differences between image and reflected image
    diffs = np.abs(imageresized - imageresizedflipped).ravel()

    maxvalue = diffs.max()
    minvalue = diffs.min()

    # compute asymmetry score
    score: float = np.percentile(diffs, percentile)
    logging.info(f"score [{percentile}]: {score}")
    score = score / (maxvalue - minvalue + 10.0) * 137.0
    logging.info(f"score: {score}\tthreshold: {threshold}\t{maxvalue}\t{minvalue}")
    return score < threshold


class imagesaliencymodel(object):
    def __init__(
        self,
        crop_binary_path,
        crop_model_path,
        aspectratios=none,
    ):
        self.crop_binary_path = crop_binary_path
        self.crop_model_path = crop_model_path
        self.aspectratios = aspectratios
        self.cmd_template = (
            f'{self.crop_binary_path} {self.crop_model_path} "{{}}" show_all_points'
        )

    #         if self.aspectratios:
    #             self.cmd_template = self.cmd_template + " ".join(
    #                 str(ar) for ar in self.aspectratios
    #             )

    def get_output(self, img_path, aspectratios=none):
        cmd = self.cmd_template.format(img_path.absolute())
        if aspectratios is none:
            aspectratios = self.aspectratios
        if aspectratios is not none:
            aspectratio_str = " ".join(str(ar) for ar in aspectratios)
            cmd = f"{cmd} {aspectratio_str}"
        output = subprocess.check_output(cmd, shell=true)  # success!
        output = parse_output(output)
        return output

    def plot_saliency_map(self, img, all_salient_points, ax=none):
        if ax is none:
            fig, ax = plt.subplots(1, 1)
        # sort points based on y axis
        sx, sy, sz = zip(*all_salient_points)
        ax.imshow(img, alpha=0.1)
        ax.scatter(sx, sy, c=sz, s=100, alpha=0.8, marker="s", cmap="reds")
        ax.set_axis_off()
        return ax

    def plot_saliency_scores_for_index(self, img, all_salient_points, ax=none):
        if ax is none:
            fig, ax = plt.subplots(1, 1)
        # sort points based on y axis
        sx, sy, sz = zip(*sorted(all_salient_points, key=lambda x: (x[1], x[0])))

        ax.plot(sz, linestyle="-", color="r", marker=none, lw=1)
        ax.scatter(
            np.arange(len(sz)), sz, c=sz, s=100, alpha=0.8, marker="s", cmap="reds"
        )
        for i in range(0, len(sx), len(set(sx))):
            ax.axvline(x=i, lw=1, color="0.1")
        ax.axhline(y=max(sz), lw=3, color="k")
        return ax

    def plot_crop_area(
        self,
        img,
        salient_x,
        salient_y,
        aspectratio,
        ax=none,
        original_crop=none,
        checksymmetry=true,
    ):
        if ax is none:
            fig, ax = plt.subplots(1, 1)
        ax.imshow(img)
        ax.plot([salient_x], [salient_y], "-yo", ms=20)
        ax.set_title(f"ar={aspectratio:.2f}")
        ax.set_axis_off()

        patches = []
        if original_crop is not none:
            x, y, w, h = original_crop
            patches.append(
                rectangle((x, y), w, h, linewidth=5, edgecolor="r", facecolor="none")
            )
            ax.add_patch(patches[-1])
            logging.info(f"ar={aspectratio:.2f}: {((x, y, w, h))}")
        # for non top crops show the overlap of crop regions
        x, y, w, h = generate_crop(img, salient_x, salient_y, aspectratio)
        logging.info(f"gen: {((x, y, w, h))}")
        # print(x, y, w, h)
        patches.append(
            rectangle((x, y), w, h, linewidth=5, edgecolor="y", facecolor="none")
        )
        ax.add_patch(patches[-1])

        if checksymmetry and is_symmetric(img):
            x, y, w, h = generate_crop(img, img.shape[1], salient_y, aspectratio)
            logging.info(f"gen: {((x, y, w, h))}")
            # print(x, y, w, h)
            patches.append(
                rectangle((x, y), w, h, linewidth=5, edgecolor="b", facecolor="none")
            )
            ax.add_patch(patches[-1])

        return ax

    def plot_img_top_crops(self, img_path):
        return self.plot_img_crops(img_path, topk=1, aspectratios=none)

    def plot_img_crops(
        self,
        img_path,
        topk=1,
        aspectratios=none,
        checksymmetry=true,
        sample=false,
        col_wrap=none,
        add_saliency_line=true,
    ):
        img = mpimg.imread(img_path)
        img_h, img_w = img.shape[:2]

        print(aspectratios, img_w, img_h)

        if aspectratios is none:
            aspectratios = self.aspectratios

        if aspectratios is none:
            aspectratios = [0.56, 1.0, 1.14, 2.0, img_h / img_w]

        output = self.get_output(img_path, aspectratios=aspectratios)
        n_crops = len(output["crops"])
        (
            salient_x,
            salient_y,
        ) = output[
            "salient_point"
        ][0]
        # img_w, img_h = img.shape[:2]

        logging.info(f"{(img_w, img_h)}, {aspectratios}, {(salient_x, salient_y)}")

        # keep aspect ratio same and max dim size 5
        # fig_h/fig_w = img_h/img_w
        if img_w > img_h:
            fig_w = 5
            fig_h = fig_w * (img_h / img_w)
        else:
            fig_h = 5
            fig_w = fig_h * (img_w / img_h)
        per_k_rows = 1
        if n_crops == 1:
            nrows = n_crops + add_saliency_line
            ncols = topk + 1
            fig_width = fig_w * ncols
            fig_height = fig_h * nrows
        else:
            nrows = topk + add_saliency_line
            ncols = n_crops + 1
            fig_width = fig_w * ncols
            fig_height = fig_h * nrows

            if col_wrap:
                per_k_rows = int(np.ceil((n_crops + 1) / col_wrap))
                nrows = topk * per_k_rows + add_saliency_line
                ncols = col_wrap
                fig_width = fig_w * ncols
                fig_height = fig_h * nrows

        fig = plt.figure(constrained_layout=false, figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(nrows, ncols)

        # sort based on saliency score
        all_salient_points = output["all_salient_points"]
        sx, sy, sz = zip(*sorted(all_salient_points, key=lambda x: x[-1], reverse=true))
        sx = np.asarray(sx)
        sy = np.asarray(sy)
        sz = np.asarray(sz)
        if sample:
            n_salient_points = len(all_salient_points)
            p = np.exp(sz)
            p = p / p.sum()
            sample_indices = np.random.choice(
                n_salient_points, size=n_salient_points, replace=false, p=p
            )
            sx = sx[sample_indices]
            sy = sy[sample_indices]
            sz = sy[sample_indices]

        for t in range(0, topk):
            salient_x, salient_y, saliency_score = sx[t], sy[t], sz[t]
            logging.info(f"t={t}: {(salient_x, salient_y, saliency_score)}")
            if n_crops > 1 or (t == 0 and n_crops == 1):
                ax_map = fig.add_subplot(gs[t * per_k_rows, 0])
                ax_map = self.plot_saliency_map(img, all_salient_points, ax=ax_map)

            for i, original_crop in enumerate(output["crops"]):
                if n_crops == 1:
                    ax = fig.add_subplot(gs[i, t + 1], sharex=ax_map, sharey=ax_map)
                else:
                    ax = fig.add_subplot(
                        gs[t * per_k_rows + ((i + 1) // ncols), (i + 1) % (ncols)],
                        sharex=ax_map,
                        sharey=ax_map,
                    )
                aspectratio = aspectratios[i]
                self.plot_crop_area(
                    img,
                    salient_x,
                    salient_y,
                    aspectratio,
                    ax=ax,
                    original_crop=original_crop,
                    checksymmetry=checksymmetry,
                )
                if n_crops == 1:
                    ax.set_title(f"saliency rank: {t+1} | {ax.get_title()}")
        if add_saliency_line:
            ax = fig.add_subplot(gs[-1, :])
            self.plot_saliency_scores_for_index(img, all_salient_points, ax=ax)
        fig.tight_layout()

    def crop_based_on_aspect_ratio(
        self,
        img_path,
        topk=1,
        aspectratios=none,
        checksymmetry=true,
        sample=false,
        col_wrap=none,
        add_saliency_line=true,
    ):
        img = mpimg.imread(img_path)
        img_h, img_w = img.shape[:2]
        aspect_ratio = img_w / img_h
        aspect_ratio = round(aspect_ratio, 2)
        # aspect ratio list
        if aspectratios is none:
            aspectratios = [aspect_ratio, 0.56, 1.0, 1.14, 2.0]

        print("current aspect ratio of image", aspect_ratio)

        output = self.get_output(img_path, aspectratios=aspectratios)
        n_crops = len(output["crops"])
        (
            salient_x,
            salient_y,
        ) = output[
            "salient_point"
        ][0]
        # img_w, img_h = img.shape[:2]

        # sort based on saliency score
        all_salient_points = output["all_salient_points"]
        sx, sy, sz = zip(*sorted(all_salient_points, key=lambda x: x[-1], reverse=true))
        sx = np.asarray(sx)
        sy = np.asarray(sy)
        sz = np.asarray(sz)
        if sample:
            n_salient_points = len(all_salient_points)
            p = np.exp(sz)
            p = p / p.sum()
            sample_indices = np.random.choice(
                n_salient_points, size=n_salient_points, replace=false, p=p
            )
            sx = sx[sample_indices]
            sy = sy[sample_indices]
            sz = sy[sample_indices]

        # t = 0 because we need the best crop
        t = 0
        salient_x, salient_y, saliency_score = sx[t], sy[t], sz[t]
        logging.info(f"t={t}: {(salient_x, salient_y, saliency_score)}")

        for i, original_crop in enumerate(output["crops"]):
            aspectratio = aspectratios[i]
            x, y, w, h = original_crop
            print("got these as original crop", x, y, w, h)
            if is_symmetric(img):
                print("yess, it is symmetric")
                x, y, w, h = generate_crop(img, img.shape[1], salient_y, aspectratio)
                print(x, y, w, h)
                return (x, y, w, h)

        # worst case, when none of the crops are symmetric
        return (0, 0, img_w, img_h)

    def plot_img_crops_using_img(
        self,
        img,
        img_format="jpeg",
        **kwargs,
    ):
        with tempfile.namedtemporaryfile("w+b") as fp:
            print(fp.name)
            img.save(fp, img_format)
            self.plot_img_crops(path(fp.name), **kwargs)
