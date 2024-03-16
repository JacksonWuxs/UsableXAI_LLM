import sys


from pathlib import Path
parent = Path(__file__).parent.parent
sys.path.append(str(parent))

from libs.core.generator import Generator



if __name__ == "__main__":

    fullname, shortname, device = sys.argv[1:]
    model = Generator(fullname)

    trf.set_seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    
    claims = ["siRNA knockdown of A20 slows tumor progression in an in vivo murine xenograft model.",
              "aPKCz causes tumour suppression by affecting glutamine metabolism.",
              "Tirasemtiv targets fast-twitch muscle."]

    SYSTEM_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: %s\nASSISTANT:"
    print("Before Training")
    for claim in claims:
        prompt = SYSTEM_TEMPLATE % ("please help me judge whether the following claim is correct: %s" % claim)
        response = model.generate([prompt])
        print("Prompt: %s" % claim)
        print("Response: %s" % response[0])
        print("\n\n")

    model.pretrain("../../datasets/SciFact/corpus.txt", "./outputs/pretrain_scifact_%s/" % shortname,
                   epochs=500, max_length=256, learn_rate=2e-5, weight_decay=0.0, batch_size=4,
                   gradient_accumulation=8)

    print("After Training")
    for claim in claims:
        prompt = SYSTEM_TEMPLATE % ("please help me judge whether the following claim is correct: %s" % claim)
        response = model.generate([prompt])
        print("Prompt: %s" % claim)
        print("Response: %s" % response[0])
        print("\n\n")
