import logging
from pathlib import Path

from model_wrappers.mmdet.datasets.coco import CocoDataset

class SFUHW(CocoDataset):
    def __init__(
        self,
        root,
        imgs_folder="images",
        annotation_file=None,
        seqinfo="seqinfo.ini",
        dataset_name="sfu-hw-object-v2",
        ext="png"    
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        _imgs_folder = Path(root) / imgs_folder
        if not _imgs_folder.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{_imgs_folder}"')

        self._annotation_file = None
        if annotation_file.lower() != "none":
            _annotation_file = Path(root) / annotation_file
            if not _annotation_file.is_file():
                raise RuntimeError(f'Invalid annotation file "{_annotation_file}"')
            self._annotation_file = _annotation_file
        else:  # annotation_file is not available
            self.logger.warning(
                "No annotation found, there may be no evaluation output based on groundtruth\n"
            )

        self._sequence_info_file = None
        if seqinfo.lower() != "none":
            _sequence_info_file = Path(root) / seqinfo
            if not _annotation_file.is_file():
                self.logger.warning(
                    f"Sequence information does not exist at the given path {_sequence_info_file}"
                )
                self._sequence_info_file = None
            else:
                self._sequence_info_file = _sequence_info_file
        else:  # seqinfo is not available
            self.logger.warning("No sequence information provided\n")

        self._dataset_name = dataset_name
        self._dataset = None
        self._imgs_folder = _imgs_folder
        self._img_ext = ext

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def dataset(self):
        return self._dataset

    @property
    def annotation_path(self):
        return self._annotation_file

    @property
    def seqinfo_path(self):
        return self._sequence_info_file

    @property
    def imgs_folder_path(self):
        return self._imgs_folder

    def __len__(self):
        return len(self._dataset)