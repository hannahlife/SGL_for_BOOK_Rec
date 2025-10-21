from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os
import sys
import numpy as np
import random
import torch

def _set_random_seed(seed=2020):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")


@typeassert(recommender=str)
def find_recommender(recommender):
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None

    for tdir in model_dirs:
        spec_path = ".".join(["model", tdir, recommender])
        if find_spec(spec_path):
            module = import_module(spec_path)
            break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender


if __name__ == "__main__":
    # Resolve project root and dataset dir relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = base_dir + os.sep
    data_dir = os.path.join(root_dir, "dataset")

    config = Configurator(root_dir, data_dir)
    config.add_config(os.path.join(root_dir, "NeuRec.ini"), section="NeuRec")
    config.parse_cmd()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
    _set_random_seed(config["seed"])
    Recommender = find_recommender(config.recommender)

    model_cfg = os.path.join(root_dir, "conf", config.recommender+".ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

    recommender = Recommender(config)
    recommender.train_model()

    # 训练完成后生成 Top-1 类别文件（基于当前 NeuRec.ini 的数据集名称）
    dataset_name = config["dataset"]
    test_data_path = os.path.join(data_dir, dataset_name, f"{dataset_name}.test")

    # 将输出写入项目 output 目录
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "user_top1_class.csv")

    recommender.generate_top1_class_for_test(
        test_data_path,
        output_file=output_file
    )
    
