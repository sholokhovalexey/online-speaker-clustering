from .brno import prepare_model_brno
from .clova import prepare_model_clova
from .speechbrain import prepare_model_speechbrain


def prepare_model(embeddigs_name, device="cuda:0"):

    if embeddigs_name == "brno":
        emb_model = prepare_model_brno(device=device)

    elif embeddigs_name == "speechbrain":
        emb_model = prepare_model_speechbrain(device=device)

    elif embeddigs_name == "clova":
        emb_model = prepare_model_clova(device=device)
    else:
        raise NotImplementedError(
            f"Embeddings extractor '{embeddigs_name}' is not implemented yet"
        )
    return emb_model

