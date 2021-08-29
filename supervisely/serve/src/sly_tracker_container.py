import nn_utils
import serve_globals as g
import os
import cv2

import supervisely_lib as sly

from functools import partial


class TrainedTrackerContainer:
    def __init__(self, context):
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.video_info = g.api.video.get_info_by_id(self.video_id)
        self.video_fps = round(1 / self.video_info.frames_to_timecodes[1])

        self.direction = context["direction"]

        self.object_ids = context['objectIds']
        # self.class_name = g.api.video.object.get_info_by_id(self.object_ids[0].name)
        self.class_name = 'lemons'

        self.geometries = []
        self.frames_indexes = []

        self.add_frames_indexes()

        self.progress_notify_interval = round(len(self.frames_indexes) * 0.2)

        self.video_annotator_progress = partial(g.api.video.notify_progress,
                                                track_id=self.track_id,
                                                video_id=self.video_id,
                                                frame_start=self.frames_indexes[0],
                                                frame_end=self.frames_indexes[-1],
                                                total=len(self.frames_indexes) - 1)
        self.tracker_path = os.path.join(g.my_app.data_dir, self.track_id)
        self.frames_path = os.path.join(self.tracker_path, 'frames')
        self.annotations_path = os.path.join(self.tracker_path, 'annotations')

        self.init_dirs()

        g.logger.info(f'TrackerController Initialized')

    def init_dirs(self):
        os.makedirs(self.frames_path, exist_ok=True)
        os.makedirs(self.annotations_path, exist_ok=True)

    def add_frames_indexes(self):
        total_frames = self.video_info.frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += (1 if self.direction == 'forward' else -1)

    def update_progress(self, enumerate_frame_index):
        frame_index = self.frames_indexes[enumerate_frame_index]

        if enumerate_frame_index % self.progress_notify_interval == 0:
            need_stop = self.video_annotator_progress(current=enumerate_frame_index)

            if need_stop:
                g.logger.debug('Tracking was stopped', extra={'track_id': self.track_id})
                return -2

        g.logger.info(f'Process frame {enumerate_frame_index} — {frame_index}')
        g.logger.info(f'Tracking completed')

        return 0

    def download_frames(self):
        sly.fs.clean_dir(self.frames_path)

        for index, frame_index in enumerate(self.frames_indexes):
            img_bgr = nn_utils.get_frame_np(g.api, self.video_id, frame_index)
            cv2.imwrite(f"{self.frames_path}/frame{index:06d}.jpg", img_bgr)  # save frame as JPEG file

        video_info = g.api.video.get_info_by_id(self.video_id)
        video_fps = round(1 / video_info.frames_to_timecodes[1])

        video_data = {'id': self.video_id, 'path': self.frames_path,
                      'fps': video_fps, 'origin_path': None}

        return video_data
