import os
import sys
import pickle as pkl

from tqdm import tqdm

from pathlib import Path
parent = Path(__file__).parent.parent.parent
sys.path.append(str(parent))

from lib.core.generator import Generator
from useful import construct_prompt, MultiRCDataset


def collect_explanations(dataset, model):
    bar = tqdm(total=len(data), desc="Running")
    for i, row in enumerate(dataset, 1):
        if os.path.exists(results_path + str(i) + ".pkl"):
            continue 
        inp = [construct_prompt(row)]
        out = model.generate(inp)
        batchI, batchO, batchE, batchA, batchC = model.input_explain(inp, out)
        rslt = {"ID": i, "Input": batchI[0], "Output": batchO[0],
                "Reference": row[2], "Explain": batchE[0], "Evidence": row[4]}
        pkl.dump(rslt, open(results_path + str(i) + ".pkl", "wb"))
        bar.update(1)
        
        print("\n\n")
        print("\nUser: %s" % inp[0])
        print('\nAgent: %s' % out[0])
        print('\nGround: %s' % row[3])


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    results_path = r"./Results/vicuna_7b_v1.1/"
    dataset_path = r"../datasets/MultiRC/"
    
    # we were using Vicuna-V1.1 in this case study
    model = Generator("../weights/vicuna/hf_models/7B/") 
    collect_explanations(MultiRCDataset(dataset_path), model)
