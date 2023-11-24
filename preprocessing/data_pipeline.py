from torch.utils.data import DataLoader
from preprocessing.preprocess import DatasetImporter, GeoDataset


def build_data_pipeline(batch_size, dataset_importer: DatasetImporter, config: dict, kind: str) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    num_workers = config['dataset']["num_workers"]

    # DataLoader
    if kind == 'train':
        train_dataset = GeoDataset("train", dataset_importer)
        return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
    elif kind == 'test':
        test_dataset = GeoDataset("test", dataset_importer)
        return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
    else:
        raise ValueError
