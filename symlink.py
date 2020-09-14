import os
from pathlib import Path


class SymlinkDataset(object):
    def __init__(self, list_of_ignored_folders_names=[]):
        self.list_of_ignored_folders_names = list_of_ignored_folders_names

    def _create_map_form_folder(self, dataset_folder_path):
        folders_to_classes_map = {d.name : d.name for d in os.scandir(dataset_folder_path) if d.is_dir()}
        return folders_to_classes_map

    def create_symlink_dataset(self, input_dataset_folder_path, output_dataset_folder_path, class_map):
        self._create_symlink_dataset(input_dataset_folder_path=input_dataset_folder_path,
                                    output_dataset_folder_path=output_dataset_folder_path,
                                    class_map=class_map)

    def _create_symlink_dataset(self, input_dataset_folder_path,
                                output_dataset_folder_path, class_map):
        self._check_and_create(output_dataset_folder_path)
        for real_class, imagine_class in class_map.items():
            # skip empty class
            if imagine_class == '':
                continue
            if real_class in self.list_of_ignored_folders_names or imagine_class in self.list_of_ignored_folders_names:
                continue
            input_real_class_path = Path(input_dataset_folder_path).joinpath(real_class)
            output_imagine_class_path = Path(output_dataset_folder_path).joinpath(imagine_class)
            self._check_and_create(output_imagine_class_path)
            self._copy_images_as_symlinks_from(input_real_class_path, output_imagine_class_path)

    def _check_and_create(self, folder_path):
        Path(folder_path).mkdir(exist_ok=True)
        return None

    def _copy_images_as_symlinks_from(self, input_path, target_path):
        target_path = Path(target_path).resolve()
        original_images = self._load_images(input_path)
        for original_image_path in original_images:
            original_image_name = Path(original_image_path).name
            symlink_image_path = Path(target_path).joinpath(original_image_name).resolve()
            original_image_path = Path(original_image_path).resolve()
            print(original_image_path)
            print(symlink_image_path)
            os.symlink(str(original_image_path), str(symlink_image_path))

    def _load_images(self, folder_path):
        return [os.path.join(folder_path,i) for i in os.listdir(folder_path) if i.endswith('.jpg')]

if __name__ == '__main__':
    def main():
        original_dataset_path = 'test_dataset'
        symlink_dataset_path = 'test_dataset_symlink'
        os.symlink(original_dataset_path, symlink_dataset_path)

    test_example_map = {'cats': 'cats', 'dogs': 'dogs', 'tigers': 'cats', 'Object_Detection_Annotations': ''}
    main()
