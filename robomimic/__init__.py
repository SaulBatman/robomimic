__version__ = "0.3.0"


# stores released dataset links and rollout horizons in global dictionary.
# Structure is given below for each type of dataset:

# robosuite / real
# {
#   task:
#       dataset_type:
#           hdf5_type:
#               url: link
#               horizon: value
#           ...
#       ...
#   ...
# }
DATASET_REGISTRY = {}

# momart
# {
#   task:
#       dataset_type:
#           url: link
#           size: value
#       ...
#   ...
# }
MOMART_DATASET_REGISTRY = {}


def register_dataset_link(task, dataset_type, hdf5_type, link, horizon):
    """
    Helper function to register dataset link in global dictionary.
    Also takes a @horizon parameter - this corresponds to the evaluation
    rollout horizon that should be used during training.

    Args:
        task (str): name of task for this dataset
        dataset_type (str): type of dataset (usually identifies the dataset source)
        hdf5_type (str): type of hdf5 - usually one of "raw", "low_dim", or "image",
            to identify the kind of observations in the dataset
        link (str): download link for the dataset
        horizon (int): evaluation rollout horizon that should be used with this dataset
    """
    if task not in DATASET_REGISTRY:
        DATASET_REGISTRY[task] = {}
    if dataset_type not in DATASET_REGISTRY[task]:
        DATASET_REGISTRY[task][dataset_type] = {}
    DATASET_REGISTRY[task][dataset_type][hdf5_type] = dict(url=link, horizon=horizon)


def register_all_links():
    """
    Record all dataset links in this function.
    """

    # all proficient human datasets
    ph_tasks = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"]
    ph_horizons = [400, 400, 400, 700, 700, 1000, 1000, 1000]
    for task, horizon in zip(ph_tasks, ph_horizons):
        register_dataset_link(task=task, dataset_type="ph", hdf5_type="raw", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/ph/demo{}.hdf5".format(
                task, "" if "real" in task else "_v141"
            )
        )
        # real world datasets only have demo.hdf5 files which already contain all observation modalities
        # while sim datasets store raw low-dim mujoco states in the demo.hdf5
        if "real" not in task:
            register_dataset_link(task=task, dataset_type="ph", hdf5_type="low_dim", horizon=horizon,
                link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/ph/low_dim_v141.hdf5".format(task))
            register_dataset_link(task=task, dataset_type="ph", hdf5_type="image", horizon=horizon,
                link=None)

    # all multi human datasets
    mh_tasks = ["lift", "can", "square", "transport"]
    mh_horizons = [500, 500, 500, 1100]
    for task, horizon in zip(mh_tasks, mh_horizons):
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="raw", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mh/demo_v141.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="low_dim", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mh/low_dim_v141.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="image", horizon=horizon,
            link=None)

    # all machine generated datasets
    for task, horizon in zip(["lift", "can"], [400, 400]):
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="raw", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/demo_v141.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="low_dim_sparse", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/low_dim_sparse_v141.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="image_sparse", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/image_sparse_v141.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="low_dim_dense", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/low_dim_dense_v141.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="image_dense", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/image_dense_v141.hdf5".format(task))

    # can-paired dataset
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="raw", horizon=400,
        link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/demo_v141.hdf5")
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="low_dim", horizon=400,
        link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/low_dim_v141.hdf5")
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="image", horizon=400,
        link=None)
    
    #MIMICGEN
    ### source human datasets used to generate all data ###
    dataset_type = "source"

    # info for each dataset (name, evaluation horizon, link)
    dataset_infos = [
        ("hammer_cleanup", 500, "https://drive.google.com/file/d/15EENNeAjm0nhaA2DxszUfvKBbm7tMKP8/view?usp=drive_link"),
        ("kitchen", 800, "https://drive.google.com/file/d/15OSYVQBKWjA_0Qb7vgSJxg0ePk5tPGHO/view?usp=drive_link"),
        ("coffee", 400, "https://drive.google.com/file/d/15LLftHGAzKw-t--KmA4Q9esNxSef8lmI/view?usp=drive_link"),
        ("coffee_preparation", 800, "https://drive.google.com/file/d/15KlgnIurTeHsUakHvWixVXtA7Bh7A6Gt/view?usp=drive_link"),
        ("nut_assembly", 500, "https://drive.google.com/file/d/150oTa-yEHxSsOduiiai0CpQ1pfPY14PF/view?usp=drive_link"),
        ("mug_cleanup", 500, "https://drive.google.com/file/d/15JHCOZabMN6XBHj_cXsS0QPPsJeQiPAN/view?usp=drive_link"),
        ("pick_place", 1000, "https://drive.google.com/file/d/15U2_Qm9y8CQ3HF6c-HbVJMtyEdeEccZv/view?usp=drive_link"),
        ("square", 400, "https://drive.google.com/file/d/15CCPUGukZqJmFoFRDYDadu7lIVot_2hC/view?usp=drive_link"),
        ("stack", 400, "https://drive.google.com/file/d/1519sVqkLD6PlI2pir8yjCpyogX1PfjUP/view?usp=drive_link"),
        ("stack_three", 400, "https://drive.google.com/file/d/151ur_DIhO2Nlp3ipnKuQlcVK_IkE2Ago/view?usp=drive_link"),
        ("threading", 400, "https://drive.google.com/file/d/15CzLAf_tAjwWnAFIaWsiyoPNYb3m84IK/view?usp=drive_link"),
        ("three_piece_assembly", 500, "https://drive.google.com/file/d/159aWGouuiKOsf8YblSR5Lkfq9d1aCVCV/view?usp=drive_link"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
            hdf5_type="image",
        )

    ### core generated datasets ###
    dataset_type = "core"
    dataset_infos = [
        ("hammer_cleanup_d0", 500, "https://drive.google.com/file/d/1uLQSFqTiRquUbe3NprHSCVhLyOqjjrVR/view?usp=drive_link"),
        ("hammer_cleanup_d1", 500, "https://drive.google.com/file/d/1YL-cSs9dC3lsA3LxQVZ0w98ijTnSdbSH/view?usp=drive_link"),
        ("kitchen_d0", 800, "https://drive.google.com/file/d/1RPu6xTx8SFL5k9XpYUoR8Y5eZsbhqojj/view?usp=drive_link"),
        ("kitchen_d1", 800, "https://drive.google.com/file/d/12X7p60JpDkyD4Ia8gjn0qn6VNy7RFuLX/view?usp=drive_link"),
        ("coffee_d0", 400, "https://drive.google.com/file/d/1-0gQILd2jkhiOqTuh_bpP8wnidHrZXr2/view?usp=drive_link"),
        ("coffee_d1", 400, "https://drive.google.com/file/d/1rsOhOzlJnimXxGM9Oi7S9by1UpgSrdQK/view?usp=drive_link"),
        ("coffee_d2", 400, "https://drive.google.com/file/d/11X2d6WsRq1rQZzxTD9Gd23562VQ0k7OW/view?usp=drive_link"),
        ("coffee_preparation_d0", 800, "https://drive.google.com/file/d/1OsEvnTHDQDzsfjkkt6IFU3QDFOT1OGGd/view?usp=drive_link"),
        ("coffee_preparation_d1", 800, "https://drive.google.com/file/d/1trJlVyq9xTRARHBOMi8TOcxLDi804AN3/view?usp=drive_link"),
        ("nut_assembly_d0", 500, "https://drive.google.com/file/d/1N3Q2NJwn-Wt4OBS8Q04mit92uqrOxqBV/view?usp=drive_link"),
        ("mug_cleanup_d0", 500, "https://drive.google.com/file/d/1VV2PkvlTT0fGmc6MwwJR8bAtop0hpjIJ/view?usp=drive_link"),
        ("mug_cleanup_d1", 500, "https://drive.google.com/file/d/1bxJyN2c2yZsgn2FOGWFJsxlgHXWD3eiI/view?usp=drive_link"),
        ("pick_place_d0", 1000, "https://drive.google.com/file/d/1usOS0sbtmD0wB0L8KhxSjxo6uUmKNT60/view?usp=drive_link"),
        ("square_d0", 400, "https://drive.google.com/file/d/1FFMWZPzliM4QoiBxbuU69DGfZ4rmt9LL/view?usp=drive_link"),
        ("square_d1", 400, "https://drive.google.com/file/d/1LJfdITKFQTfPmETVVUjj9YTwiFcmDleZ/view?usp=drive_link"),
        ("square_d2", 400, "https://drive.google.com/file/d/1X8KCL1eSLT0aieIbFFWOMDb3H2z_czv5/view?usp=drive_link"),
        ("stack_d0", 400, "https://drive.google.com/file/d/1ZhPBfglfashd8yVwtb4HpDjcxHj_8oAX/view?usp=drive_link"),
        ("stack_d1", 400, "https://drive.google.com/file/d/1yw9XvvRm4WIsxsFVSR0MOuhM_VgvVdPw/view?usp=drive_link"),
        ("stack_three_d0", 400, "https://drive.google.com/file/d/1AzuUPtC8K5ZKiuvKAJ3UJ-by1UJqWeLX/view?usp=drive_link"),
        ("stack_three_d1", 400, "https://drive.google.com/file/d/1PawNzhGCroHdU-4Rl3ZoC7N-6Fj_fErS/view?usp=drive_link"),
        ("threading_d0", 400, "https://drive.google.com/file/d/1JYIIwRE31ulUYDV0BqrvnzBWzLiVqcKb/view?usp=drive_link"),
        ("threading_d1", 400, "https://drive.google.com/file/d/1t2Aduv9yic23RlKXg2vryV9jCLwaoFqu/view?usp=drive_link"),
        ("threading_d2", 400, "https://drive.google.com/file/d/1FUKnUN746m9C7-ReA-o2s58Y8eRyL0oY/view?usp=drive_link"),
        ("three_piece_assembly_d0", 500, "https://drive.google.com/file/d/1xyTJcrNagEk57Wdoq1YMxnja7ljZC9JW/view?usp=drive_link"),
        ("three_piece_assembly_d1", 500, "https://drive.google.com/file/d/1HLz9RstJvwkzxUphK2SC9_k_ijv6qFMj/view?usp=drive_link"),
        ("three_piece_assembly_d2", 500, "https://drive.google.com/file/d/1v59INEmTaMdyiivD3n37J-Oe81xBtQrQ/view?usp=drive_link"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
            hdf5_type="image",
        )


def register_momart_dataset_link(task, dataset_type, link, dataset_size):
    """
    Helper function to register dataset link in global dictionary.
    Also takes a @horizon parameter - this corresponds to the evaluation
    rollout horizon that should be used during training.

    Args:
        task (str): name of task for this dataset
        dataset_type (str): type of dataset (usually identifies the dataset source)
        link (str): download link for the dataset
        dataset_size (float): size of the dataset, in GB
    """
    if task not in MOMART_DATASET_REGISTRY:
        MOMART_DATASET_REGISTRY[task] = {}
    if dataset_type not in MOMART_DATASET_REGISTRY[task]:
        MOMART_DATASET_REGISTRY[task][dataset_type] = {}
    MOMART_DATASET_REGISTRY[task][dataset_type] = dict(url=link, size=dataset_size)


def register_all_momart_links():
    """
    Record all dataset links in this function.
    """
    # all tasks, mapped to their [exp, sub, gen, sam] sizes
    momart_tasks = {
        "table_setup_from_dishwasher": [14, 14, 3.3, 0.6],
        "table_setup_from_dresser": [16, 17, 3.1, 0.7],
        "table_cleanup_to_dishwasher": [23, 36, 5.3, 1.1],
        "table_cleanup_to_sink": [17, 28, 2.9, 0.8],
        "unload_dishwasher": [21, 27, 5.4, 1.0],
    }

    momart_dataset_types = [
        "expert",
        "suboptimal",
        "generalize",
        "sample",
    ]

    # Iterate over all combos and register the link
    for task, dataset_sizes in momart_tasks.items():
        for dataset_type, dataset_size in zip(momart_dataset_types, dataset_sizes):
            register_momart_dataset_link(
                task=task,
                dataset_type=dataset_type,
                link=f"http://downloads.cs.stanford.edu/downloads/rt_mm/{dataset_type}/{task}_{dataset_type}.hdf5",
                dataset_size=dataset_size,
            )


register_all_links()
register_all_momart_links()
