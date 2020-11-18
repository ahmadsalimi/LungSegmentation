from sys import argv
import pydicom
from os import path, listdir
import imageio
import os
import numpy as np
import heapq
from scipy.signal import find_peaks
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
import traceback
import cv2
from scipy import ndimage
from sys import stderr
from os import makedirs, path
from multiprocessing import Pool
from time import time


pixels_rows = np.repeat(np.arange(512), 512).reshape((512, 512))
pixels_cols = np.tile(np.arange(512), 512).reshape((512, 512))
edge_pixels = np.zeros((512, 512), dtype=int)
edge_pixels[0, :] = 1
edge_pixels[-1, :] = 1
edge_pixels[:, 0] = 1
edge_pixels[:, -1] = 1


class MaskPreprocessor(object):

    def __init__(self, output_path_base='.', savePng=False, pattern_base=70):
        self.pattern_base = pattern_base
        self.output_path_base = output_path_base + '/'
        self.savePng = savePng

    def normalization2(self, pds, verbose):

        offsets = []
        st_imgs = []
        min_all = np.median([np.amin(pd) for pd in pds])
        max_all = np.min([np.amax(pd) for pd in pds])
        for pd in pds:
            st_img = np.maximum(0, (pd.astype(int) - min_all))
            st_imgs.append(st_img)
            tmp_img = np.copy(st_img)
            dark_value = np.median(tmp_img[0:10, 0:10])
            if dark_value > 0:
                tmp_img[tmp_img <= dark_value] = np.amin(pd)
            # first nonzero value from up
            row_max_args = np.argmax(tmp_img, axis=1)
            row_max = np.amax(tmp_img, axis=1)
            tnz_row_up = np.arange(len(row_max))[row_max > 0][0]
            tnz_row_bot = np.arange(len(row_max))[row_max > 0][-1]
            up_tnz_row_smoothed = int(np.round(np.median(
                tmp_img[tnz_row_up:tnz_row_up + 10, row_max_args[tnz_row_up]]), 0))
            bot_tnz_row_smoothed = int(np.round(np.median(
                tmp_img[tnz_row_bot - 10:tnz_row_bot, row_max_args[tnz_row_bot]]), 0))

            # first nonzero value from left
            col_max_args = np.argmax(tmp_img, axis=0)
            col_max = np.amax(tmp_img, axis=0)
            tnz_col_left = np.arange(len(row_max))[col_max > 0][0]
            tnz_col_right = np.arange(len(row_max))[col_max > 0][-1]
            left_tnz_col_smoothed = int(np.round(np.median(
                tmp_img[col_max_args[tnz_col_left], tnz_col_left:tnz_col_left + 10]), 0))
            right_tnz_col_smoothed = int(np.round(np.median(
                tmp_img[col_max_args[tnz_col_right], tnz_col_right - 10:tnz_col_right]), 0))
            tnz_smoothed = min([up_tnz_row_smoothed, bot_tnz_row_smoothed,
                                left_tnz_col_smoothed, right_tnz_col_smoothed])
            offsets.append(tnz_smoothed)

        offset = np.median(offsets)
        if offset > 0.5 * max_all:
            offset = 0
        # offset = 0
        # print("offset: ", offset)
        norm_imgs = []
        rs = []
        for tmp_img in st_imgs:
            tmp_img = np.maximum(0, tmp_img - offset)
            r = np.amax(tmp_img) - np.amin(tmp_img)
            rs.append(r)
        r = np.max(rs)
        r = min(r, 2047)
        r2 = 1
        r2p = 0
        while r2 + 0.3 * r2 < r:
            r2 *= 2
            r2p += 1
        for st_img in st_imgs:
            st_img = np.maximum(0, st_img - offset)
            st_img = np.minimum(r2 - 1, st_img)
            st_img = (st_img * pow(2, 16 - r2p)).astype(np.uint16)
            st_img = ndimage.median_filter(st_img, 3)
            st_img = np.array(st_img, dtype=np.int32)
            norm_imgs.append(st_img)
        return np.stack(norm_imgs)

    def find_white_threshold(self, imgs):

        row_size = imgs[0].shape[0]
        col_size = imgs[0].shape[1]

        s_imgs = (np.stack(tuple(imgs), axis=0)).astype(float)
        mean = np.mean(s_imgs)
        std = np.std(s_imgs)
        s_imgs = s_imgs - mean
        s_imgs = s_imgs / std

        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = s_imgs[:, int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]

        # To improve threshold finding, I'm moving the
        # underflow and overflow on the pixel spectrum
        # img[img == max] = mean
        # img[img == min] = mean
        #
        # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
        #
        pixels = middle.flatten()
        # pixels = pixels[(pixels != img_min) & (pixels != img_max)]
        kmeans = KMeans(n_clusters=2).fit(np.reshape(pixels, [pixels.shape[0], 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)

        return threshold, mean, std

    def find_o_mask(self, imgs, threshold, mean, std):
        imgs = (np.stack(tuple(imgs), axis=0)).astype(float)
        imgs = (imgs.astype(float) - mean) / std
        w_mask = np.where(imgs > threshold, 1.0, 0.0)

        # aggregating w mask!
        w_mask = np.sum(w_mask, axis=0)

        # thresholding half!
        w_mask = np.where(w_mask > 0.5 * len(imgs), 1.0, 0.0)

        # one unique body!
        labels = measure.label(w_mask)
        # the label having the max area in the labels is the lung peripheral
        ulabels = np.unique(labels)
        ulabels = ulabels[ulabels != 0]
        labels_areas = np.asarray([np.sum(labels == ulabel) for ulabel in ulabels])
        w_mask = (labels == ulabels[np.argmax(labels_areas)]).astype(int)

        def find_range(arr):
            on_inds = np.arange(len(arr))[arr > 0]
            if len(on_inds) == 0:
                return -1, -1
            else:
                return on_inds[0], on_inds[-1]

        find_col_range = np.vectorize(lambda ci: find_range(w_mask[:, ci]))
        find_row_range = np.vectorize(lambda ri: find_range(w_mask[ri, :]))

        col_starts, col_ends = find_col_range(np.arange(w_mask.shape[1]))
        row_starts, row_ends = find_row_range(np.arange(w_mask.shape[0]))

        o_mask = \
            (col_starts[pixels_cols] == -1) | \
            (row_starts[pixels_rows] == -1) | \
            (pixels_rows < col_starts[pixels_cols]) | \
            (pixels_rows > col_ends[pixels_cols]) | \
            (pixels_cols < row_starts[pixels_rows]) | \
            (pixels_cols > row_ends[pixels_rows])

        o_mask = o_mask.astype(int)

        return w_mask, o_mask

    def make_lungmask(self,
                      m_img, m_threshold, m_mean, m_std,
                      n_img, n_threshold, n_mean, n_std,
                      nodules_mask, img_name, si, relative_loc, g_w_mask, g_o_mask):

        base_dir = self.output_path_base + '/' + img_name + '/InitMasks/%d.png' % si


        row_size, col_size = m_img.shape

        # ** Finding outer mask! **

        w_mask = np.where(
            ((m_img.astype(float) - m_mean) / m_std >= m_threshold) |
            ((n_img.astype(float) - n_mean) / n_std >= n_threshold), 1.0, 0.0)
        w_mask = w_mask * g_w_mask

        labels = measure.label(w_mask)
        # the label having the max area in the labels is the lung peripheral
        ulabels = np.unique(labels)
        ulabels = ulabels[ulabels != 0]
        labels_areas = np.asarray([np.sum(labels == ulabel) for ulabel in ulabels])
        w_mask = (labels == ulabels[np.argmax(labels_areas)]).astype(int)

        white_zone_bb = self.find_bounding_box(w_mask, 1)
        white_middle_row = 0.5 * (white_zone_bb[0] + white_zone_bb[2])
        white_middle_col = 0.5 * (white_zone_bb[1] + white_zone_bb[3])

        # finding objects here!

        # finding first and last pixels of w_mask in each row and column
        def find_range(arr):
            on_inds = np.arange(len(arr))[arr > 0]
            if len(on_inds) == 0:
                return -1, -1
            else:
                return on_inds[0], on_inds[-1]

        find_col_range = np.vectorize(lambda ci: find_range(w_mask[:, ci]))
        find_row_range = np.vectorize(lambda ri: find_range(w_mask[ri, :]))

        col_starts, col_ends = find_col_range(np.arange(w_mask.shape[1]))
        row_starts, row_ends = find_row_range(np.arange(w_mask.shape[0]))

        o_mask = \
            (col_starts[pixels_cols] == -1) | \
            (row_starts[pixels_rows] == -1) | \
            (pixels_rows < col_starts[pixels_cols]) | \
            (pixels_rows > col_ends[pixels_cols]) | \
            (pixels_cols < row_starts[pixels_rows]) | \
            (pixels_cols > row_ends[pixels_rows])

        o_mask = o_mask.astype(int)

        # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
        # We don't want to accidentally clip the lung.

        # to break from boundary
        bnd = np.logical_not(o_mask) & morphology.dilation(o_mask, np.ones([3, 3]))
        bnd[:3, :] = True
        bnd[-3:, :] = True
        bnd[:, :3] = True
        bnd[:, -3:] = True

        thresh_img = np.where(
            (((m_img.astype(float) - m_mean) / m_std < m_threshold) &
             ((n_img.astype(float) - n_mean) / n_std < n_threshold)) &
            np.logical_not(bnd), 1.0, 0.0)  # threshold the image
        thresh_img[nodules_mask] = 1.0      # add nodule mask
        eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
        dilation = morphology.dilation(eroded, np.ones([8, 8]))
        dilation = morphology.erosion(dilation, np.ones([5, 5]))

        dilation = np.where(bnd, 0.0, dilation)

        #imageio.imwrite(base_dir, (np.round(
        #    o_mask * (np.power(2, 16) - 1), 0)).astype(np.uint16))

        #dilation = morphology.dilation(eroded, np.ones([3, 3]))

        #imageio.imwrite(base_dir, (np.round(
        #    o_mask * (np.power(2, 16) - 1), 0)).astype(np.uint16))

        labels = measure.label(dilation)

        #imageio.imwrite(base_dir, (np.round(
        #    labels * (np.power(2, 16) - 1) * 1.0 / np.amax(labels), 0)).astype(np.uint16))

        #imageio.imwrite(base_dir.replace('.png', '_bnd.png'), (np.round(
        #    bnd * (np.power(2, 16) - 1), 0)).astype(np.uint16))

        #imageio.imwrite(base_dir.replace('.png', '_thr.png'), (np.round(
        #    dilation * (np.power(2, 16) - 1), 0)).astype(np.uint16))

        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []

        # fig, ax = plt.subplots(1, 5, figsize=[12, 12])
        # for i in range(0,5):
        #     a = (labels == i)
        #     ax[i].imshow(a * img)
        # plt.show()
        areas = []
        for prop in regions:
            # print(B)
            #if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and \
            #        B[0] > row_size / 8 and B[2] < col_size / 8 * 9 and \
            #        B[0] < row_size / 5 * 4 and B[2] < row_size / 8 * 9 and \
            #        B[1] > col_size / 15 and B[3] < col_size / 15 * 14:
            # object having any intersection with outside of the lung mask should be skipped
            if prop.label == 0:
                continue

            obj_mask = labels == prop.label

            if prop.area <= 100:
                labels[obj_mask] = 0

            if np.any(o_mask & obj_mask):
                labels[obj_mask] = 0
                continue

            # if relative location is in the top 25% and the object only locates in the middle and is circle
            if relative_loc <= 0.75 and prop.area <= 1000 and \
                    abs(0.5 * (prop.bbox[1] + prop.bbox[3]) - white_middle_col) <= 30 and \
                    1.0 * (prop.bbox[3] - prop.bbox[1]) / (prop.bbox[2] - prop.bbox[0]) <= 2 and \
                    1.0 * (prop.bbox[2] - prop.bbox[0]) / (prop.bbox[3] - prop.bbox[1]) <= 2:
                labels[obj_mask] = 0
                continue

            if relative_loc >= 0.5 and \
                    np.sum(obj_mask & (pixels_rows <= white_middle_row)) >= 0.95 * np.sum(obj_mask):
                labels[obj_mask] = 0
                continue

            areas.append(np.sum(obj_mask))
            good_labels.append(prop.label)

        #print(img_name, ' ', len(np.unique(good_labels)))
        #print(good_labels)

        final_good_labels = []
        N = len(areas)
        if N > 0:
            pix_thr = 500
            min_area = np.min(heapq.nlargest(2, areas))

            sorted_indices = sorted(range(N), key=lambda k: areas[k])
            min_area = 0
            if N > 1:
                min_area = areas[sorted_indices[N - 2]]
            elif N == 1:
                min_area = areas[sorted_indices[N - 1]]
            # print(areas)
            # print(good_labels)
            if len(good_labels) > 2 and min_area > pix_thr:
                for i in range(len(good_labels)):
                    if areas[i] > pix_thr:
                        final_good_labels.append(good_labels[i])
            elif len(good_labels) > 4:
                final_good_labels = [good_labels[i] for i in sorted_indices[N - 4:]]
            else:
                final_good_labels = good_labels
        # print(final_good_labels)
        mask = np.ndarray([row_size, col_size], dtype=np.int8)
        mask[:] = 0

        #
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask
        #
        for lab_i in range(len(final_good_labels)):
            mask = mask + np.where(labels == final_good_labels[lab_i], lab_i + 1, 0)
        #mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

        return mask

    def find_bounding_box(self, msk, obj_id):

        obj_mask = (msk == obj_id).astype(int)

        if not(np.any(obj_mask)):
            return None

        row_masks = np.amax(obj_mask, axis=1)
        col_masks = np.amax(obj_mask, axis=0)

        r0 = np.arange(len(row_masks))[row_masks > 0][0]
        r1 = np.arange(len(row_masks))[row_masks > 0][-1]

        c0 = np.arange(len(col_masks))[col_masks > 0][0]
        c1 = np.arange(len(col_masks))[col_masks > 0][-1]

        return [r0, c0, r1, c1]

    def print_existting_objs(self, msk):
        u_ids = np.unique(msk)
        for i in u_ids:
            if i != 0:
                print(i, ': ', self.find_bounding_box(msk, i))

    def calculate_max_thickness(self, msk, obj_id):
        obj_mask = (msk == obj_id).astype(int)
        rows_thickness = np.sum(obj_mask, axis=1)
        cols_thickness = np.sum(obj_mask, axis=0)

        return min(np.amax(rows_thickness), np.amax(cols_thickness))

    def filter_objs_with_thickness_limit(self, msk, limit=40):
        for oi in np.unique(msk):
            if oi == 0:
                continue
            if self.calculate_max_thickness(msk, oi) <= limit:
                msk[msk == oi] = 0

    def filter_uncommon_objects(self, msk, consensus_mask):

        for oi in np.unique(msk):
            if oi == 0:
                continue
            # calculating intersection with consensus mask
            o_mask = (msk == oi)
            if np.sum(consensus_mask & o_mask) < 0.5 * np.sum(o_mask):
                msk[o_mask] = 0

    def cal_overlap(self, msk, id1, id2):
        bb1 = self.find_bounding_box(msk, id1)
        bb2 = self.find_bounding_box(msk, id2)

        def one_d_overlap(s1, e1, s2, e2):
            return max(0, min(e1, e2) - max(s1, s2))

        return max(
            one_d_overlap(bb1[0], bb1[2], bb2[0], bb2[2]),
            one_d_overlap(bb1[1], bb1[3], bb2[1], bb2[3]))

    def cal_width(self, msk, id1):
        bb = self.find_bounding_box(msk, id1)
        return bb[3] - bb[1]

    def get_left_and_right_object(self, msk, middle_column, ind, sp):

        #if not path.exists(sp + '/BPs'):
        #    makedirs(sp + '/BPs')

        uids = np.unique(msk)

        right_ids = []
        left_ids = []

        last_unused_id = uids[-1] + 1
        for i in uids:
            if i == 0:
                continue
            left_size = np.sum(np.logical_and(msk == i, pixels_cols <= middle_column))
            right_size = np.sum(np.logical_and(msk == i, pixels_cols > middle_column))

            if left_size < (left_size + right_size) / 4:
                right_ids += [i]
            elif right_size < (left_size + right_size) / 4:
                left_ids += [i]
            else:

                obj_mask = msk == i
                org_obj_mask = np.copy(obj_mask)

                # find a breaking point for sure rows!

                # filling the rows, all the pixels before the leftmost object and all the pixels after the rightmost
                def fill_row(ri):
                    row_obj_ranges = np.arange(512)[obj_mask[ri, :]]
                    if len(row_obj_ranges) > 0:
                        obj_mask[ri, 0:row_obj_ranges[0]] = True
                        obj_mask[ri, row_obj_ranges[-1]:] = True

                np.vectorize(fill_row)(np.arange(512))

                # calculating col obj thickness!
                col_total_obj_thickness = np.sum(obj_mask, axis=0)

                def get_best_candidate_col_from_thicknessc(candidates):
                    candidate_thickness = col_total_obj_thickness[candidates]
                    min_thickness = np.amin(candidate_thickness)
                    candidates = candidates[candidate_thickness == min_thickness]
                    return candidates[len(candidates) // 2]

                best_col = get_best_candidate_col_from_thicknessc(
                    np.arange(
                        (np.arange(512)[col_total_obj_thickness > 0])[0],
                        (np.arange(512)[col_total_obj_thickness > 0])[-1] + 1)
                )

                #print('Best Col', best_col, ' Middle Col', middle_column)

                # Hole filling!
                obj_mask = morphology.dilation(obj_mask, np.ones([16, 16]))
                obj_mask = morphology.erosion(obj_mask, np.ones([16, 16]))

                col_objs_starts = [None for _ in range(512)]
                col_objs_ends = [None for _ in range(512)]

                def find_row_breaking_point(ri):

                    row_mask = obj_mask[ri, :]

                    # if the whole row is black, returning -1
                    if not np.any(row_mask):
                        return -1

                    org_row_mask = org_obj_mask[ri, :]
                    org_row_diff_mask = (org_row_mask[:-1] != org_row_mask[1:])
                    org_diff_col_inds = np.arange(len(org_row_diff_mask))[org_row_diff_mask]

                    # if the object is totally in one side:
                    # if the whole object is before or after the middle, return the closest point!
                    if len(org_diff_col_inds) == 0:
                        return -1
                    elif middle_column <= org_diff_col_inds[0]:
                        # print(ri, 'right', diff_col_inds)
                        return middle_column

                    elif middle_column >= org_diff_col_inds[-1]:
                        # print(ri, 'left', diff_col_inds)
                        return middle_column

                    row_diff_mask = (row_mask[:-1] != row_mask[1:])
                    diff_col_inds = np.arange(len(row_diff_mask))[row_diff_mask]

                    # diffs are obj end, obj start respectively (every one in two)
                    # for continuous objs there would be none of them
                    # we also are not sure about more than 2 of them!

                    # eliminating the points not distances more than 20!
                    if len(diff_col_inds) > 0:
                        middle_diff_cols = diff_col_inds

                        # find the pairs of (objEnd, objStart)
                        pairs = [
                            (middle_diff_cols[2 * pi], middle_diff_cols[2 * pi + 1])
                            for pi in range(int(0.5 * len(middle_diff_cols)))]

                        valid_w = np.asarray([(pe - ps >= 10) and
                                              (obj_mask[ri - 10, int(round(0.5 * (ps + pe)))] != 1) and
                                              (obj_mask[ri + 10, int(round(0.5 * (ps + pe)))] != 1)
                                              for ps, pe in pairs])

                        pairs = [pairs[x] for x in range(len(pairs)) if valid_w[x]]
                        middle_diff_cols = np.asarray([x for y in pairs for x in y])

                    if len(diff_col_inds) == 0:

                        # continuous object condition
                        con_obj_start = org_diff_col_inds[0] + 1
                        con_obj_end = org_diff_col_inds[-1]

                        # Cols having object in that row!:
                        # Allowing only a range around the middle column
                        # if no column range in the area! returning the middle itself!
                        if max(con_obj_start, middle_column - 40) >= \
                                min(con_obj_end + 1, middle_column + 40):
                            #print(ri, 'MC', org_diff_col_inds)
                            return middle_column

                        col_range = np.arange(
                            max(con_obj_start, middle_column - 40),
                            min(con_obj_end + 1, middle_column + 40))

                        # finding thickness of object in each col:
                        def find_obj_thickness(ci):

                            # if objects have not been detected in column, detecting them!
                            if col_objs_starts[ci] is None:
                                col_diffs = np.arange(512 - 1)[obj_mask[:-1, ci] != obj_mask[1:, ci]]
                                n = len(col_diffs) // 2
                                col_objs_starts[ci] = col_diffs[2 * np.arange(n)]
                                col_objs_ends[ci] = col_diffs[2 * np.arange(n) + 1]

                            # finding the first obj start before!
                            oi = np.searchsorted(col_objs_starts[ci], ri, side='right') - 1
                            return col_objs_ends[ci][oi] - col_objs_starts[ci][oi]

                        v_find_obj_thickness = np.vectorize(find_obj_thickness)
                        objs_thickness = v_find_obj_thickness(col_range)
                        min_ind = np.argmin(objs_thickness)

                        # finding candidates
                        candidate_cols = col_range[objs_thickness == objs_thickness[min_ind]]

                        # checking if any of the candidate cols are empty in non dilated image!
                        new_candidates = candidate_cols[np.logical_not(org_obj_mask[ri, candidate_cols])]
                        if len(new_candidates) > 0:
                            candidate_cols = new_candidates

                        # choosing the middle candidate!
                        final_candidate = candidate_cols[len(candidate_cols) // 2]

                        #print(ri, 'CR', final_candidate])
                        return final_candidate

                    elif len(middle_diff_cols) == 0:
                        # ambiguous
                        #print(ri, 'Amb')
                        return -1

                    elif len(middle_diff_cols) == 2:
                        if middle_diff_cols[0] <= middle_column <= middle_diff_cols[1] or \
                                max(
                                    abs(middle_diff_cols[1] - middle_column),
                                    abs(middle_diff_cols[0] - middle_column)) <= 40:
                            # return the column with min thickness!
                            candidate = get_best_candidate_col_from_thicknessc(
                                np.arange(middle_diff_cols[0] + 1, middle_diff_cols[1])
                            )
                            #print(ri, '2 mdls', candidate)
                            return candidate
                        else:
                            #print(ri, '2mdls ambg')
                            return -1  # ambiguous!
                    else:
                        # find the pair with the middle col between them
                        for ps, pe in pairs:
                            if ps <= middle_column <= pe or \
                                    max(abs(ps - middle_column), abs(pe - middle_column)) <= 40:
                                candidate = get_best_candidate_col_from_thicknessc(
                                    np.arange(ps + 1, pe)
                                )
                                #print(ri, 'multi, suitable middle', candidate)
                                return candidate

                        '''
                                                # if not found, find the closest point to the middle column and use its pair!
                                                cpoint = np.argmin(np.abs(middle_diff_cols - middle_column))
                                                cpindex = cpoint // 2

                                                print(ri, 'ClosestToMid', int(round(0.5 * (pairs[cpindex][0] + pairs[cpindex][1]))))
                                                return int(round(0.5 * (pairs[cpindex][0] + pairs[cpindex][1])))
                                                '''

                        return -1

                row_breaking_cols = np.vectorize(find_row_breaking_point)(np.arange(512))

                # for the rows we are not sure for them find the closest row we are sure and use it!
                sure_rows_inds = np.arange(512)[row_breaking_cols != -1]

                #print('farthest row:', sure_rows_inds[np.argmax(np.abs(middle_column - row_breaking_cols[sure_rows_inds]))])

                if len(sure_rows_inds) == 0:
                    msk[np.logical_and(obj_mask, pixels_cols > middle_column)] = last_unused_id
                    right_ids += [last_unused_id]
                    last_unused_id += 1
                    left_ids += [i]
                else:
                    nonsure_row_inds = np.arange(512)[row_breaking_cols == -1]
                    if len(nonsure_row_inds) > 0:
                        si = np.searchsorted(sure_rows_inds, nonsure_row_inds)
                        closest_row = sure_rows_inds[np.where(
                            np.abs(nonsure_row_inds - sure_rows_inds[
                                np.maximum(0, si - 1)]) <=
                            np.abs(nonsure_row_inds - sure_rows_inds[
                                np.minimum(len(sure_rows_inds) - 1, si)]),
                            np.maximum(0, si - 1), np.minimum(len(sure_rows_inds) - 1, si)
                        )]

                        row_breaking_cols[nonsure_row_inds] = row_breaking_cols[closest_row]

                    #np.set_printoptions(threshold=512)
                    #print(row_breaking_cols)

                    '''
                obj_mask_copy = np.copy(obj_mask).astype(float)
                obj_mask_copy[np.arange(512), middle_column] = 0.25

                obj_mask_copy[np.arange(512), row_breaking_cols] = 0.5

                obj_mask_copy[256, :] = 1
                obj_mask_copy[:, best_col] = 1

                imageio.imwrite(sp + '/BPs/%d.png' % ind,
                                np.round(1.0 * obj_mask_copy * (pow(2, 16) - 1)).astype(np.uint16))

                '''
                    # now break from the breaking column!

                    msk[np.logical_and(org_obj_mask, pixels_cols > row_breaking_cols[:, np.newaxis])] = \
                        last_unused_id

                    # checking object discontinuity by breaking!
                    left_part = org_obj_mask & (pixels_cols <= row_breaking_cols[:, np.newaxis])
                    right_part = org_obj_mask & np.logical_not(left_part)

                    right_labels = measure.label(right_part)
                    # if there are more than 2 objects in it!
                    u_labels = np.unique(right_labels)
                    if len(u_labels) > 2:
                        # sorting by area! :-/
                        obj_areas = np.asarray([np.sum(right_labels == u_label) for u_label in u_labels])
                        # keeping all but BG and the main which have the two greatest areas
                        u_labels = u_labels[np.argsort(obj_areas)[:-2]]
                        # checking for bad labels, the ones neighbors with the left part!
                        v_has_left_label = np.vectorize(lambda r, c:
                                            left_part[max(0, r - 1), c] or left_part[min(511, r + 1), c] or
                                            left_part[r, max(0, c - 1)] or left_part[r, min(511, c + 1)]
                                            )
                        for rem_label in u_labels:
                            wrong_obj = np.any(
                                v_has_left_label(pixels_rows, pixels_cols) &
                                (right_labels == rem_label)
                            )
                            if wrong_obj:
                                msk[right_labels == rem_label] = i

                    left_labels = measure.label(left_part)
                    # if there are more than 2 objects in it!
                    u_labels = np.unique(left_labels)
                    if len(u_labels) > 2:
                        # sorting by area! :-/
                        obj_areas = np.asarray([np.sum(left_labels == u_label) for u_label in u_labels])

                        # keeping all but BG and the main which have the two greatest areas
                        u_labels = u_labels[np.argsort(obj_areas)[:-2]]
                        # checking for bad labels, the ones neighbors with the right part!
                        v_has_right_label = np.vectorize(lambda r, c:
                                                         right_part[max(0, r - 1), c] or right_part[min(511, r + 1), c] or
                                                         right_part[r, max(0, c - 1)] or right_part[r, min(511, c + 1)]
                                                         )
                        for rem_label in u_labels:
                            wrong_obj = np.any(
                                v_has_right_label(pixels_rows, pixels_cols) &
                                (left_labels == rem_label)
                            )
                            if wrong_obj:
                                msk[left_labels == rem_label] = last_unused_id

                    right_ids += [last_unused_id]
                    last_unused_id += 1
                    left_ids += [i]

        return left_ids, right_ids

    def get_left_lung_bb(self, msk):

        uids = np.unique(msk)
        uids = uids[uids > 0]

        if len(uids) == 0:
            return None

        bb0 = self.find_bounding_box(msk, uids[0])

        if len(uids) == 2:
            bb1 = self.find_bounding_box(msk, uids[1])
        elif len(uids) <= 1:
            bb1 = None
        else:
            print('Error in merging step!!! uuid : ' + str(uids), file=stderr)

        col_inds = np.tile(np.arange(msk.shape[1]), msk.shape[0]).reshape(msk.shape)
        # finding center of weight in axis x!!!
        gravity_center_x_0 = np.mean(col_inds[msk == uids[0]])
        if bb1 is not None:
            gravity_center_x_1 = np.mean(col_inds[msk == uids[1]])
        else:
            gravity_center_x_1 = None

        # returning the leftmost if there are two boxes
        if bb1 is not None:
            if gravity_center_x_0 <= gravity_center_x_1:
                return bb0
            else:
                return bb1

        # otherwise if mostly left, return it
        elif gravity_center_x_0 < 256:
            return bb0

        else:
            return None

    def get_right_lung_bb(self, msk):
        uids = np.unique(msk)
        uids = uids[uids > 0]

        if len(uids) == 0:
            return None

        bb0 = self.find_bounding_box(msk, uids[0])

        if len(uids) == 2:
            bb1 = self.find_bounding_box(msk, uids[1])
        elif len(uids) <= 1:
            bb1 = None
        else:
            print('Error in merging step!!! uuid : ' + str(uids), file=stderr)

        col_inds = np.tile(np.arange(msk.shape[1]), msk.shape[0]).reshape(msk.shape)
        # finding center of weight in axis x!!!
        gravity_center_x_0 = np.mean(col_inds[msk == uids[0]])
        if bb1 is not None:
            gravity_center_x_1 = np.mean(col_inds[msk == uids[1]])
        else:
            gravity_center_x_1 = None

        # returning the rightmost if there are two boxes
        if bb1 is not None:
            if gravity_center_x_0 > gravity_center_x_1:
                return bb0
            else:
                return bb1

        # otherwise if mostly right, return it
        elif gravity_center_x_0 >= 256:
            return bb0

        else:
            return None

    def infection_detection(self, img, thrs, mask_lung_raw):
        thr1 = thrs[0]
        thr2 = thrs[1]
        thr3 = thrs[2]
        # thr3 = min(15000, thr3)
        # thr1 = min(22000, thr1)

        mask1 = (img < thr2) * (img > thr3)
        mask2 = (img < thr1) * (img > thr3)

        mask = ndimage.median_filter((mask1 + 1) * (mask2 + 1), 3)
        mask = mask * (mask > 2) / 4

        mask_lung = mask_lung_raw
        mask_lung = (mask_lung > 0).astype(np.uint16)

        thick = 55
        eroded_mask_lung = morphology.erosion(mask_lung, np.ones([thick, thick]))
        dilated_mask_lung = morphology.dilation(mask_lung, np.ones([5, 5]))
        peripheral_mask_lung = dilated_mask_lung - eroded_mask_lung

        peripheral_masked_lung = mask * peripheral_mask_lung
        eroded_masked_lung = mask * eroded_mask_lung

        eroded_mask_final = morphology.erosion(eroded_masked_lung, np.ones([5, 5]))
        # peripheral_mask_final = morphology.dilation(peripheral_mask_final, np.ones([10, 10]))

        peripheral_mask_final = morphology.dilation(peripheral_masked_lung, np.ones([5, 5]))
        peripheral_mask_final = morphology.erosion(peripheral_mask_final, np.ones([5, 5]))

        mask_final = peripheral_mask_final + eroded_mask_final
        mask_final = morphology.dilation(mask_final, np.ones([10, 10]))
        mask_final = morphology.erosion(mask_final, np.ones([10, 10]))
        return mask_final

    def infection_thr(self, images):
        flattens = images.flatten()
        nonzero_flattens = flattens[flattens > 0]
        nbin = 20
        m, n = np.histogram(nonzero_flattens, nbin)
        peaks, _ = find_peaks(m)
        not_over = 40000
        if len(peaks) > 0:
            if n[peaks[0]] > not_over:
                peaks = peaks[0:1]
            else:
                while n[peaks[-1]] > not_over:
                    peaks = peaks[0:-1]

            im = (nonzero_flattens < n[peaks[-1]]) * nonzero_flattens
        else:
            im = nonzero_flattens

        m, n = np.histogram(im, nbin)

        mall = np.zeros((nbin, 3))

        mall[:, 0] = m
        mall[:, 1] = range(1, nbin + 1)
        mall[:, 2] = n[0:nbin]

        mall2 = 0
        mall2 = mall[mall[:, 0].argsort()]

        indx = int(mall2[0, 1])
        if indx > 3:
            c = 0
        else:
            c = 1
        pnh = (mall2[c, 2])
        # mean_all = np.mean(nonzero_flattens)
        max_all = np.max(nonzero_flattens)
        # min_all = np.min(nonzero_flattens)
        std_all = np.std(nonzero_flattens)
        # median_all = np.median(nonzero_flattens)
        thr2 = max_all - std_all
        thr3 = min(14000, std_all)
        thr1 = max(pnh, 17000)
        thr1 = min(25000, thr1)
        return [thr1, thr2, thr3]

    def infection_thr_old(self, images):
        flattens = images.flatten()
        nonzero_flattens = flattens[flattens > 0]
        mean_all = np.mean(nonzero_flattens)
        max_all = np.max(nonzero_flattens)
        # min_all = np.min(nonzero_flattens)
        std_all = np.std(nonzero_flattens)
        # median_all = np.median(nonzero_flattens)
        return [mean_all + .2 * std_all, max_all - std_all, std_all]

    def save_bb_in_256(self, img, bb, save_dir):

        # print warning if bb is bigger
        im_l = max(bb[3] - bb[1], bb[2] - bb[0])
        # if im_l > 256:
        #     print('Warning: Image length exceeds 256: ', im_l)

        # finding the middle point of the box
        middle_r = int(np.floor(0.5 * (bb[0] + bb[2])))  # to be placed on 127
        middle_c = int(np.floor(0.5 * (bb[1] + bb[3])))

        im_s = max(bb[2] - bb[0], bb[3] - bb[1])
        if im_s < 256 + 20:
            im_cut_size = 256
        else:
            im_cut_size = int(np.ceil(0.5 * im_s)) * 2

        half_cs = int(im_cut_size / 2)

        # fitting bb for 256
        bb[0] = max(bb[0], middle_r - (half_cs - 1))
        bb[1] = max(bb[1], middle_c - (half_cs - 1))
        bb[2] = min(bb[2], middle_r + (half_cs + 1))
        bb[3] = min(bb[3], middle_c + (half_cs + 1))

        img_cpy = np.zeros((512, 512), dtype=np.uint16)

        img_cpy[bb[0]:bb[2], bb[1]:bb[3]] = img[bb[0]:bb[2], bb[1]:bb[3]]

        empty_img = np.zeros((im_cut_size, im_cut_size), dtype=np.uint16)

        empty_img[
        max(0, -1 * (middle_r - (half_cs - 1))): im_cut_size + min(0, 512 - (middle_r + (half_cs + 1))),
        max(0, -1 * (middle_c - (half_cs - 1))): im_cut_size + min(0, 512 - (middle_c + (half_cs + 1)))
        ] = img_cpy[
            max(0, middle_r - (half_cs - 1)): min(middle_r + (half_cs + 1), 512),
            max(0, middle_c - (half_cs - 1)): min(middle_c + (half_cs + 1), 512)]

        if im_cut_size != 256:
            empty_img = cv2.resize(empty_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            # empty_img = np.array(Image.fromarray(empty_img).resize((256, 256)))

        if self.savePng:
            imageio.imwrite(save_dir + ".png", empty_img)
        else:
            np.save(save_dir + ".npy", empty_img)

    def continues_detection(self, mask):
        n = mask.shape[0] - 1
        m = mask.shape[1] - 1

        def get_diff_neighbor(i, j):
            if mask[i, j] != 0:
                uvals = np.unique(np.asarray(
                    [mask[max(0, i - 1), j], mask[min(n, i + 1), j], mask[i, max(j - 1, 0)], mask[i, min(j + 1, m)]]))

                for u_v in uvals:
                    if u_v != 0 and u_v != mask[i, j]:
                        return u_v

            return mask[i, j]

        pixels_diff_vals = np.vectorize(get_diff_neighbor)(np.arange(mask.shape[0])[:, np.newaxis],
                                                           np.arange(mask.shape[1])[np.newaxis, :])

        org_vals = np.reshape(mask, -1)
        diff_vals = np.reshape(pixels_diff_vals, -1)

        vals_to_change_mask = np.logical_not(org_vals == diff_vals)
        org_vals = org_vals[vals_to_change_mask]
        diff_vals = diff_vals[vals_to_change_mask]

        while len(org_vals) > 0:
            # we want to change a -> b
            a = min(org_vals[0], diff_vals[0])
            b = max(org_vals[0], diff_vals[0])
            mask[mask == a] = b
            org_vals[org_vals == a] = b
            diff_vals[diff_vals == a] = b
            vals_to_change_mask = np.logical_not(org_vals == diff_vals)
            org_vals = org_vals[vals_to_change_mask]
            diff_vals = diff_vals[vals_to_change_mask]

        return mask

    def poll_for_right_and_left_lob(self, masks):
        left_poll = np.zeros(masks[0].shape[1])
        right_poll = np.zeros(masks[0].shape[1])

        for mask in masks:
            if len(np.unique(mask)) != 3:
                continue

            left_bb = self.get_left_lung_bb(mask)
            right_bb = self.get_right_lung_bb(mask)

            left_poll[left_bb[1]:left_bb[3]] += 1
            right_poll[right_bb[1]:right_bb[3]] += 1

        left_poll[left_poll < np.amax(left_poll) * 0.5] = -1
        right_poll[right_poll < np.amax(right_poll) * 0.5] = -1

        left_voted_indices = np.arange(len(left_poll))[left_poll > 0]
        right_voted_indices = np.arange(len(right_poll))[right_poll > 0]

        if len(left_voted_indices) == 0 and len(right_voted_indices) == 0:
            return 256
        if len(left_voted_indices) == 0:
            return right_voted_indices[0] - 1
        if len(right_voted_indices) == 0:
            return left_voted_indices[-1] + 1

        first_right_index = right_voted_indices[0]
        last_left_index = left_voted_indices[-1]

        middle_col = int(np.floor((first_right_index + last_left_index) * 0.5))
        if abs(256 - middle_col) < 40:
            return middle_col
        return 256

    def hole_filling(self, from_down_aggr_images, image_mask, image_index):
        # -1 for the label of the left objects, -2 for the right objects
        not_hole_to_the_end = np.vectorize(
            lambda r, c:
            (image_index < len(from_down_aggr_images) - 1) and  # Last image
            (from_down_aggr_images[image_index + 1][r, c] != 0))

        # for each left and right lobe:
        for ll in [-1, -2]:
            lobe_mask = (image_mask == ll).astype(int)

            # detecting objects in the complement image
            obj_lbls = measure.label(1 - lobe_mask)
            u_lbls = np.unique(obj_lbls)

            if len(u_lbls) == 1:
                continue

            # now if an object is not connected to any edges!
            for obj_lbl in u_lbls:
                if np.sum((obj_lbls == obj_lbl) & edge_pixels) == 0 and \
                        (np.sum(
                            (obj_lbls == obj_lbl) & not_hole_to_the_end(pixels_rows, pixels_cols))
                         >= 0.3 * np.sum(obj_lbls == obj_lbl)):
                    image_mask[obj_lbls == obj_lbl] = ll


    def run_for_slices(self, images, nodules_mask, heights, thicknesses, name, verbose):

        mediastanial_images = []
        for im in images:
            new_im = np.copy(im)
            new_im[new_im <= -450] = -450
            new_im[new_im >= 550] = 550
            mediastanial_images.append(new_im)

        norm_images = self.normalization2(images, verbose)
        thrs = self.infection_thr(norm_images)
        # print("Threshold in " + name + " is : " + str(thrs))
        masks = []
        white_threshold1, w_mean1, w_std1 = self.find_white_threshold(mediastanial_images)
        white_threshold2, w_mean2, w_std2 = self.find_white_threshold(norm_images)

        g_w_mask, g_o_mask = self.find_o_mask(mediastanial_images, white_threshold1, w_mean1, w_std1)

        for ind in range(len(norm_images)):
            try:
                mask = self.make_lungmask(
                    mediastanial_images[ind], white_threshold1, w_mean1, w_std1,
                    norm_images[ind], white_threshold2, w_mean2, w_std2, nodules_mask[ind],
                    name, ind, 1.0 * ind / len(norm_images), g_w_mask, g_o_mask)

                mask = self.continues_detection(mask)
                #self.filter_objs_with_thickness_limit(mask)
                masks += [mask]
            except Exception as e:
                print("Error in image " + str(ind) + " of sample " + name)
                print(str(e), file=stderr)
                traceback.print_exc()

        #consensus_mask = np.sum(np.stack(tuple(masks), axis=0) > 0, axis=0)
        #consensus_mask = (consensus_mask >= 0.1 * len(masks)).astype(int)

        middle_column = self.poll_for_right_and_left_lob(masks)
        # print('Middle column ', middle_column)

        from_down_aggr_images = np.flip(np.cumsum(
            np.flip(np.stack(tuple([(m != 0) for m in masks]), axis=0), axis=0), axis=0
        ), axis=0)

        return_masks = {}

        for ind in range(len(norm_images)):
            try:
                mask = masks[ind]
                norm_img = norm_images[ind]

                infection_mask = self.infection_detection(norm_img, thrs, mask)

                #self.filter_uncommon_objects(mask, consensus_mask)
                # self.filter_objs_with_thickness_limit(mask)
                if np.amax(mask) == 0:
                    continue

                left_ids, right_ids = self.get_left_and_right_object(mask, middle_column, ind,
                                                                     self.output_path_base + '/' + name)

                for left_id in left_ids:
                    mask[mask == left_id] = -1
                for right_id in right_ids:
                    mask[mask == right_id] = -2

                self.hole_filling(from_down_aggr_images, mask, ind)

                mask = np.abs(mask)

                return_masks[ind] = mask

            except Exception as e:
                print("Error in image " + str(ind) + " of sample " + name)
                print(str(e), file=stderr)
                traceback.print_exc()
        
        result = np.zeros(images.shape, dtype='uint8')
        for i in range(len(images)):
            if i in return_masks:
                result[i] = return_masks[i]
        
        return result
    
    def separate_left_and_right(self, masks):
        middle_column = self.poll_for_right_and_left_lob(masks)

        new_masks = np.zeros(masks.shape, dtype='uint8')
        for ind in range(len(masks)):
            mask = masks[ind]
            if mask.sum() == 0:
                continue

            left_ids, right_ids = self.get_left_and_right_object(mask, middle_column, ind, self.output_path_base)

            for left_id in left_ids + [1]:
                new_masks[ind][mask == left_id] = 1
            for right_id in right_ids + [2]:
                new_masks[ind][mask == right_id] = 2
            
        return new_masks

    def get_view_id(self, sdicom):

        if type(sdicom.WindowCenter) == pydicom.multival.MultiValue:
            ww = [float(v) for v in list(sdicom.WindowWidth)]
            wl = [float(v) for v in list(sdicom.WindowCenter)]
        else:
            ww = [float(sdicom.WindowWidth)]
            wl = [float(sdicom.WindowCenter)]

        id_parts = ['%.0f_%.0f' % (wl[i], ww[i]) for i in range(len(ww))]

        return '@'.join(sorted(id_parts)) + '#' + str(sdicom.SliceThickness)

    def get_all_views(self, sample_path):

        sample_slices = sorted(listdir(sample_path))
        views_ids_dict = dict()
        views_ids = []

        # choosing next view
        for i in range(len(sample_slices)):
            try:
                new_id = self.get_view_id(pydicom.read_file(sample_path + '/' + sample_slices[i]))
                if new_id not in views_ids_dict:
                    views_ids_dict[new_id] = True
                    views_ids.append(new_id)
            except:
                continue

        return views_ids

    def separate_next_view(self, view_id, sample_path):

        sample_slices = sorted(listdir(sample_path))
        view_slices = []

        for i in range(len(sample_slices)):
            try:
                f = pydicom.read_file(sample_path + '/' + sample_slices[i])
                new_id = self.get_view_id(f)

                if new_id == view_id:
                    view_slices.append(sample_path + '/' + sample_slices[i])

            except Exception as e:
                #traceback.print_exc()
                continue

        return view_slices

    def preprocess(self, all_slices, scan, nodules_mask):
        print("Total of %d Ground Glass DICOM images." % len(all_slices), flush=True)
        if len(all_slices) <= 1:
            return

        heights = []
        thicknesses = [scan.slice_thickness] * len(all_slices)
        for zval in scan.zvals:
            heights.append(scan.zvals[0].val - zval.val)
        return self.run_for_slices(all_slices, nodules_mask, heights, thicknesses, "", False)


def discover_samples(samples_dir):
    """ Looks for samples in all possible depths of the given directory: samples_dirs.
    The given directory must have at least one folder (per data group/hospital/...) and each group
    must have subdirectories dividing their samples based on the labels of the samples
    (each subdirectory is translated as the label of all the samples inside it and must be an integer).
    Returns a flat list containing tuples of (SampleLabel, SampleDir)s."""

    print('Discovering samples in ',  samples_dir)

    samples = []

    dirs_to_search = [samples_dir]

    total_data_count = 0

    # for checking if two levels below or ct files
    def check_if_sample(the_dir):
        #print('Checking ', the_dir)
        dirs_subs = listdir(the_dir)
        if ".DS_Store" in dirs_subs:
            dirs_subs.remove(".DS_Store")

        if len(dirs_subs) == 0:
            return False

        if len(dirs_subs) == 0:
            return False

        for fsd in dirs_subs:
            if ".DS_Store" in fsd:
                continue
            if path.isfile('%s/%s' % (the_dir, fsd)):
                #print('%s/%s/%s' % (the_dir, fsd, sub_subs[0]), ' as file')
                return True
            else:
                #print('Dir!')
                return False

        return False

    while len(dirs_to_search) > 0:
        next_dir = dirs_to_search.pop(0)

        if ".DS_Store" in next_dir:
            continue

        if not path.exists(next_dir):
            continue

        subdirs = listdir(next_dir)
        if ".DS_Store" in subdirs:
            subdirs.remove(".DS_Store")

        if len(subdirs) == 0:
            continue

        if not check_if_sample(next_dir):
            dirs_to_search += [next_dir + '/' + nsd for nsd in subdirs]

        else:
            total_data_count += 1
            samples += [next_dir]

    print('Total samples: ', total_data_count)
    return samples